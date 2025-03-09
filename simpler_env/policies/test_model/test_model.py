import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from simpler_env.utils.value_norm import ValueNorm

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class MLPDiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPDiagGaussian, self).__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.logstd.expand_as(mean)
        std = logstd.exp()
        var = std ** 2

        with torch.no_grad():
            sample = torch.normal(mean, std)
        sample_logit = -((sample - mean) ** 2) / (2 * var) - logstd - math.log(math.sqrt(2 * math.pi))
        # -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        mode = mean
        mode_logit = - logstd - math.log(math.sqrt(2 * math.pi))
        raw_prob = torch.cat((mean, logstd), dim=-1)

        return sample, sample_logit, mode, mode_logit, raw_prob

    def evaluate_actions(self, x, actions):
        mean = self.fc_mean(x)
        logstd = self.logstd.expand_as(mean)
        std = logstd.exp()
        var = std ** 2

        log_prob = -((actions - mean) ** 2) / (2 * var) - logstd - math.log(math.sqrt(2 * math.pi))
        entropy = torch.mean(0.5 + 0.5 * (math.log(2 * math.pi) + logstd), dim=-1, keepdim=True)
        # 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
        raw_prob = torch.cat((mean, logstd), dim=-1)

        return log_prob, entropy, raw_prob

class MLPActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPActor, self).__init__()

        self.base = MLPNetwork(input_dim, 64)
        self.act = MLPDiagGaussian(64, output_dim)


    def forward(self, x):
        features = self.base(x)

        sample, sample_logit, mode, mode_logit, raw_prob = self.act(features)

        return sample, sample_logit, mode, mode_logit, raw_prob

    def evaluate_actions(self, x, actions):
        features = self.base(x)

        log_prob, entropy, logit = self.act.evaluate_actions(features, actions)

        return log_prob, entropy, logit

class MLPCritic(nn.Module):
    def __init__(self, input_dim):
        super(MLPCritic, self).__init__()

        self.base = MLPNetwork(input_dim, 1)

    def forward(self, x):
        return self.base(x)

class MLPPPOPolicy:
    def __init__(self, input_dim, action_dim):
        self.actor = MLPActor(input_dim, action_dim)
        self.critic = MLPCritic(input_dim)
        self.value_norm = ValueNorm()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def get(self, x, deterministic=False):
        sample, sample_logit, mode, mode_logit, raw_prob = self.actor(x)
        value = self.critic(x)
        if deterministic:
            return value, mode, mode_logit
        else:
            return value, sample, sample_logit

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        sample, sample_logit, mode, mode_logit, raw_prob = self.actor(x)
        if deterministic:
            return mode, mode_logit
        else:
            return sample, sample_logit

    def evaluate_actions(self, x, actions):
        log_prob, entropy, logit = self.actor.evaluate_actions(x, actions)
        return log_prob, entropy, logit

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.value_norm.to(device)
        return self

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        self.value_norm.eval()

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        self.value_norm.train()

class MLPPPO:
    def __init__(self, policy, device):
        self.policy = policy
        self.ppo_epoch = 1
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = 0.01
        self.ppo_huber_delta = 10.0
        self.tpdv = dict(device=device, dtype=torch.float32)

    def train_step(self, batch):
        obs, actions, value_preds, returns, masks, old_logits, advantages = batch
        obs = torch.tensor(obs, dtype=torch.float32).to(**self.tpdv)
        actions = torch.tensor(actions, dtype=torch.float32).to(**self.tpdv)
        value_preds = torch.tensor(value_preds, dtype=torch.float32).to(**self.tpdv)
        returns = torch.tensor(returns, dtype=torch.float32).to(**self.tpdv)
        masks = torch.tensor(masks, dtype=torch.float32).to(**self.tpdv)
        old_logits = torch.tensor(old_logits, dtype=torch.float32).to(**self.tpdv)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(**self.tpdv)

        returns_norm = self.policy.value_norm(returns) # [B, 1]

        # Policy loss
        logits, entropy, _ = self.policy.evaluate_actions(obs, actions)
        ratio = torch.exp(logits - old_logits)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Value loss
        values = self.policy.get_value(obs)

        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss = value_loss.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss

        self.policy.actor_optimizer.zero_grad()
        self.policy.critic_optimizer.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.ppo_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.ppo_grad_norm)

        self.policy.actor_optimizer.step()
        self.policy.critic_optimizer.step()

        self.policy.value_norm.update(returns)

        info = dict(
            loss = loss.item(),
            policy_loss = policy_loss.item(),
            value_loss = value_loss.item(),
            entropy_loss = entropy_loss.item(),
            actor_grad_norm = actor_grad_norm.item(),
            critic_grad_norm = critic_grad_norm.item(),
            ratio = ratio.mean().item(),
        )

        return info

    def train(self, buffer):
        train_info = defaultdict(lambda: [])

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for batch in data_generator:
                info = self.train_step(batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info


