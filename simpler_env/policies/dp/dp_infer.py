import os
import sys
import torch
import pickle
import imageio
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
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
        saved_model_path: str = "openvla/openvla-7b",
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
        self.usages = ['obs']

        policy_config = {
            'lr': 1e-5,
            'num_images': len(self.cameras) * len(self.usages),
            'action_dim': 10,
            'observation_horizon': 1,
            'action_horizon': 1,
            'prediction_horizon': 20,

            'global_obs_dim': 10,
            'num_inference_timesteps': 10,
            'ema_power': 0.75,
            'vq': False,
        }
    
        self.policy = DiffusionPolicy(policy_config)
        self.policy.deserialize(torch.load(saved_model_path))
        self.policy.eval()
        self.policy.cuda()

    def process_action(self, action):
        """
            unnormalize the action to the original scale, prepare for the env step.
            input: action: (10,), float, np
            output: action: (10,), float, np
        """
        action = action * np.expand_dims(self.pose_gripper_scale, axis=0) + np.expand_dims(self.pose_gripper_mean, axis=0)
        return action

    def process_data(self, image_list, proprio_state):
        """
            process the data for diffusion policy model. 
            input:  image_list: [ M * (h, w, c)],  uint8, np (M is the number of cameras)
                    proprio_state: (10,), float, np
            output: image_data: (M, c, h, w),  normalized in [0,1], float, np
                    qpos_data: (10)
        """
        all_cam_images = np.stack(image_list, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        try:
            k, c, h, w = image_data.shape
            transformations = [
                # transforms.CenterCrop((int(h * 0.95), int(w * 0.95))),
                transforms.RandomCrop((int(h * 0.95), int(w * 0.95))),
                # transforms.Resize((240, 320), antialias=True),
                transforms.Resize((224, 224), antialias=True),
            ]
            for transform in transformations:
                image_data = transform(image_data)
                # print(front.shape, goal.shape)
        except Exception as e:
            print(e)

        image_data = image_data / 255.0

        # qpos = np.array([0.], dtype=float)
        proprio_state = np.array(proprio_state)
        proprio_state = (proprio_state - self.proprio_gripper_mean) / self.proprio_gripper_scale
        qpos_data = torch.from_numpy(proprio_state).float()

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
            image_list.append(obs['sensor_data'][cam]['rgb'].squeeze(0).to(torch.uint8))

        pose:Pose = env.agent.ee_pose_at_robot_base
        self.pose_at_obs = pose.to_transformation_matrix().squeeze(0)

        pose_mat = rotation_conversions.quaternion_to_matrix(pose.q) # pose_mat = quat2mat(pose.q)
        pose_mat_6 = pose_mat[:, :2].reshape(-1).numpy()
        proprio_state = np.concatenate(
            [
                pose.p.numpy(),
                pose_mat_6,
                np.array([obs["gripper_width"]]),
            ]
        )
        image_data, qpos_data = self.process_data(image_list, proprio_state)
        image_data, qpos_data = image_data.cuda().unsqueeze(0), qpos_data.cuda().unsqueeze(0)
        import pdb;pdb.set_trace()
        # print(image_data.shape, qpos_data.shape)

        pred_actions = self.policy(qpos_data, image_data).squeeze().cpu()
        actions = self.process_action(pred_actions)

        return None, actions

    def reset(self, task_description: str) -> None:
        pass
