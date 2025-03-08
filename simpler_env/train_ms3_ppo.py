import random
import signal
from typing import Annotated
import torch
import numpy as np
import tyro
from dataclasses import dataclass
from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c


@dataclass
class Args:
    """
    This is a script to evaluate policies on real2sim environments. Example command to run:

    XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
        --model="octo-small" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes 192 --num-envs 64
    """

    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    num_envs: int = 32
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    # env
    episode_len: int = 5


class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args
        policy_setup = "widowx_bridge"

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # env
        self.env = SimlerWrapper(self.args.env_id, self.args.num_envs, policy_setup, self.args.seed)

        # policy
        from simpler_env.policies.test_model.test_model import MLPPPOPolicy, MLPPPO
        self.device = torch.device("cuda")
        self.policy = MLPPPOPolicy(3 * 480 * 640, 7)
        self.policy.to(self.device)
        self.alg = MLPPPO(self.policy, self.device)

        print()

        # buffer
        self.buffer = SeparatedReplayBuffer(
            obs_dim=3 * 480 * 640,
            act_dim=7,
            num_env=self.args.num_envs,
            ep_len=self.args.episode_len
        )

    @torch.no_grad()
    def collect(self):
        self.policy.prep_training()

        obs = self.buffer.obs[self.buffer.step]
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        value, action, logit = self.policy.get(obs)

        return value, action, logit

    def insert(self, data):
        obs_img, actions, action_log_probs, value_preds, rewards, terminated, truncated = data
        masks = 1.0 - (terminated | truncated).to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        actions = actions.cpu().numpy()
        action_log_probs = action_log_probs.cpu().numpy()
        value_preds = value_preds.cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, action_log_probs, value_preds, rewards, masks)

    def compute_and_train(self):
        self.policy.prep_rollout()

        obs = torch.tensor(self.buffer.obs[-1], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            next_value = self.policy.get_value(obs)
        next_value = next_value.cpu().numpy()

        self.buffer.compute_returns(next_value, self.policy.value_norm)

        infos = self.alg.train(self.buffer)
        infos["reward_mean"] = np.mean(self.buffer.rewards)
        infos["mask_mean"] = np.mean(self.buffer.masks)

        return infos

    def run(self):
        for ep in range(100):
            obs_img, info = self.env.reset()
            self.buffer.warmup(obs_img.cpu().numpy())

            for step in range(self.args.episode_len):
                value, action, logit = self.collect()
                obs_img, reward, terminated, truncated, info = self.env.step(action)

                data = (obs_img, action, logit, value, reward, terminated, truncated)
                self.insert(data)

                # print
                print({k: v.to(torch.float32).mean().tolist() for k, v in info.items()})

            infos = self.compute_and_train()
            print(infos)


if __name__ == "__main__":
    args = tyro.cli(Args)
    runner = Runner(args)
    runner.run()
