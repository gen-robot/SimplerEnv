import os
import sys
import torch
import pickle
import imageio
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
from mani_skill.utils import common, gym_utils
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry import rotation_conversions
from simpler_env.policies.dp.dp_modules.policy import DiffusionPolicy
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from simpler_env.policies.dp.dp_modules.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

class DPInference:
    def __init__(
        self,
        obs_normalize_params_path: str,
        saved_model_path: str,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.obs_normalize_params_path = obs_normalize_params_path
        self.OBS_NORMALIZE_PARAMS = pickle.load(open(self.obs_normalize_params_path, "rb"))
        self.pose_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["pose", "gripper_width"]
            ]
        )
        self.pose_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["pose", "gripper_width"]
            ]
        )
        self.proprio_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )
        self.proprio_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )

        self.cameras = ['3rd_view_camera']

        self.policy_config = {
            'lr': 1e-5,
            'num_images': len(self.cameras),
            'action_dim': 10,
            'observation_horizon': 1,
            'action_horizon': 1,
            'prediction_horizon': 8, # used in training

            'global_obs_dim': 10,
            'num_inference_timesteps': 10,
            'ema_power': 0.75,
            'vq': False,
        }
    
        self.policy = DiffusionPolicy(self.policy_config)
        self.policy.deserialize(torch.load(saved_model_path))
        self.policy.eval()
        self.policy.cuda()

    def process_action(self, action):
        """
        input: action: (B, 10) or (10,), float, np
        output: action: (B, 10) or (10,), float, np
        """
        action = np.asarray(action)
        if action.ndim == 1:
            action = action[None, :]
        
        action = action * self.pose_gripper_scale[None, :] + self.pose_gripper_mean[None, :]
        
        if action.shape[0] == 1:
            action = action.squeeze(0)
        
        return action

    def process_data(self, image_list, proprio_state):
        """
        Args:
            image_list: (M, B, H, W, C) list or array, M is the number of cameras
            proprio_state: (B, 10) or (10,), np.ndarray
        Returns:
            image_data: (B, M, C, H, W), torch.float32
            qpos_data: (B, 10) or (10,), torch.float32
        """
        image_list = np.asarray(image_list)  # (M, B, H, W, C)
        M, B, H, W, C = image_list.shape

        image_data = torch.from_numpy(image_list)  # (M, B, H, W, C)
        image_data = image_data.permute(1, 0, 4, 2, 3)  # (B, M, C, H, W)
        image_data = image_data.view(B * M, C, H, W)

        try:
            transformations = [
                # transforms.CenterCrop((int(H * 0.95), int(W * 0.95))),
                transforms.RandomCrop((int(H * 0.95), int(W * 0.95))),
                # transforms.Resize((240, 320), antialias=True),
                transforms.Resize((224, 224), antialias=True),
            ]
            for transform in transformations:
                image_data = transform(image_data)
                # print(front.shape, goal.shape)
        except Exception as e:
            print(e)

        image_data = image_data.view(B, M, C, 224, 224)
        image_data = image_data.float() / 255.0  # [0,1]

        proprio_state = np.asarray(proprio_state)
        if proprio_state.ndim == 1:
            proprio_state = proprio_state[None, :]  # (1, 10)
        proprio_state = (proprio_state - self.proprio_gripper_mean[None, :]) / self.proprio_gripper_scale[None, :]
        qpos_data = torch.from_numpy(proprio_state).float()

        if qpos_data.shape[0] == 1:
            qpos_data = qpos_data.squeeze(0)

        return image_data, qpos_data

    def step(
        self, env: BaseEnv, images: torch.Tensor, task_description: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """

        obs = env.get_obs()
        image_list = []
        for cam in self.cameras:
            image_list.append(obs['sensor_data'][cam]['rgb'].squeeze(0).to(torch.uint8).cpu().numpy())

        pose:Pose = env.agent.ee_pose_at_robot_base
        self.pose_at_obs = pose.to_transformation_matrix().cpu().numpy()
        pose_mat = rotation_conversions.quaternion_to_matrix(pose.q) # pose_mat = quat2mat(pose.q)
        pose_mat_6 = pose_mat[:, :2].reshape(pose_mat.shape[0],-1).cpu().numpy()        
        gripper_width = gym_utils.inv_scale_action(
                env.agent.robot.get_qpos()[:,-1], env.agent.controller.configs['gripper'].lower, env.agent.controller.configs['gripper'].upper
            )
        proprio_state = np.concatenate(
            [
                pose.p.cpu().numpy(),
                pose_mat_6,
                gripper_width.unsqueeze(-1).cpu().numpy(),
            ],
            axis = 1,
        )

        image_data, qpos_data = self.process_data(image_list, proprio_state)
        image_data, qpos_data = image_data.cuda(), qpos_data.cuda()
        # print(image_data.shape, qpos_data.shape)

        pred_actions = self.policy(qpos_data, image_data).squeeze().cpu()
        actions = self.process_action(pred_actions)

        return None, actions

    def reset(self, task_description: str) -> None:
        pass
