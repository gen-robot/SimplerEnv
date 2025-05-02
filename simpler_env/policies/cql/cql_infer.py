import d3rlpy
import os
import torch
import numpy as np

class CQLInference:
    def __init__(
        self,
        obs_normalize_params_path: str = "/home/bingwen/Documents/temp/rl/normalization_params.npz",
        saved_model_path: str = "/home/bingwen/Documents/temp/rl/cql_model.pt",
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # dataset parameters
        self.normalization_params = np.load(obs_normalize_params_path)
        self.observation_mean = self.normalization_params['observation_mean']
        self.observation_std = self.normalization_params['observation_std']
        self.action_mean = self.normalization_params['action_mean']
        self.action_std = self.normalization_params['action_std']

        self.observation_shape = (57600,)  # (480/4) * (640/4) * 3
        self.action_size = 8 # xyz(3) + quat(4) + gripper(1) 
        # Load model
        self.policy = d3rlpy.algos.CQLConfig().create(device='cuda:0')
        # # manually set observation shape and action size
        self.policy.create_impl(self.observation_shape, self.action_size)
        # load pretrained model
        self.policy.load_model(saved_model_path)

    def denormalize_action(self, action):
        """
        input: action: (B, 8), float, np
        output: action: (B, 8), float, np
        """
        action = np.asarray(action)
        action = action * self.action_std + self.action_mean
        return action

    def process_normalize_obs(self, images):
        assert len(images.shape) == 4 and images.shape[-3:] == (480, 640, 3), "input images shape error"
        B = images.shape[0]
        images = images[:, ::4, ::4, :]
        observation = images.reshape(B, -1)  # [B, 57600]
        normalized_observation = (observation - self.observation_mean) / self.observation_std  # [B, 57600]
        return normalized_observation

    def step(
        self, images: torch.Tensor, task_description: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
        # inference
        normalized_observation = self.process_normalize_obs(images)
        raw_actions = self.policy.predict(normalized_observation)
        actions = self.denormalize_action(raw_actions)
        return raw_actions, actions

    def reset(self, task_description: str) -> None:
        pass

if __name__ == "__main__":
    cql = CQLInference()
    images = np.random.rand(5, 480, 640, 3).astype(np.float32)
    raw_actions, actions = cql.step(images, None)
    print(actions)