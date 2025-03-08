import torch
import numpy as np


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, obs_dim, act_dim, num_env, ep_len, gamma=0.99, gae_lambda=0.95):
        self.ep_len = ep_len
        self.num_env = num_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.buffer_minibatch = -1

        self.obs = np.zeros((self.ep_len + 1, self.num_env, obs_dim), dtype=np.float32)
        self.value_preds = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.returns = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.actions = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.float32)
        self.action_log_probs = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.masks = np.ones((self.ep_len + 1, self.num_env, 1), dtype=np.float32)

        self.step = 0

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        self.step = (self.step + 1) % self.ep_len

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()

    def warmup(self, obs):
        self.obs[0] = obs
        self.masks[0] = 1.0

        self.step = 0

    def compute_returns(self, next_value, value_norm):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            vt1 = torch.tensor(self.value_preds[step + 1], dtype=torch.float32).to(value_norm.running_mean.device)
            vt1 = value_norm.denormalize(vt1).cpu().numpy()
            vt = torch.tensor(self.value_preds[step], dtype=torch.float32).to(value_norm.running_mean.device)
            vt = value_norm.denormalize(vt).cpu().numpy()

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt

    def feed_forward_generator(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = n_rollout_threads * episode_length

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        # calc adv
        advantages = self.returns - self.value_preds[:-1]
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            yield (obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)
