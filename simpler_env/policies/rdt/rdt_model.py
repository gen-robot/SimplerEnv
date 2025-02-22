from typing import Optional, Sequence
import matplotlib.pyplot as plt
from transforms3d.euler import euler2axangle
import cv2 as cv
import os
import numpy as np
import torch
import peft
import yaml
import json
from PIL import Image
from torchvision import transforms
from collections import deque
from copy import deepcopy
from typing import Dict, List
from third_party.rdt.constants import *
from third_party.rdt.configs import ROBOT_INDICES, ROBOT_CAMERA_NAMES
from third_party.rdt.models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from third_party.rdt.models.multimodal_encoder.t5_encoder import T5Embedder
from third_party.rdt.models.rdt_runner import RDTRunner
from simpler_env.policies.rdt.rdt_actor import RDTActor, dict_apply
from simpler_env.policies.rdt.rdt_aloha_model import (
    create_model as create_RDT_model, 
    RoboticDiffusionTransformerModel,
)

from mani_skill.utils.structs.articulation import Articulation
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.agents.utils import (
    flatten_action_spaces,
    get_active_joint_indices,
    get_joints_by_names,
)
from scipy.spatial.transform import Rotation

# reset(), step(), visualize_epoch()
class RDTInference(RDTActor):
    def __init__(self, *args, text_embedding_path=None, **kwargs): 
        kwargs.setdefault("robot_name", "widowx_bridge")
        kwargs.setdefault("action_scale", 1)
        self.robot_name = kwargs["robot_name"]
        self.action_scale = kwargs["action_scale"]
        self.text_embedding_path = text_embedding_path
        self.last_action = None
        super().__init__(*args, **kwargs)


    def get_env(self,env):
        self.env = env

    def update_obs_window(self, obs):
        if self.obs_window is None:
            self.obs_window = deque(maxlen=self.n_frames)
            self.obs_window.append(
                {
                    'qpos': None,
                    'images':
                        {
                            cam_name: None 
                                for cam_name in self.camera_names
                        }
                }
            )

        # two arm in maniskill3
        if self.robot_name in ['mobile_aloha_v2','mobile_aloha']:
            self.obs_window.append(
                {
                    'qpos': torch.concat([obs['agent']['qpos_l'], obs['agent']['qpos_r']]),
                    'images':
                        {
                            cam_name: obs['sensor_data'][cam_name]['rgb'].permute(2, 0, 1)
                                for cam_name in self.camera_names
                        }
                }
            )
        # single arm in maniskill3
        elif self.robot_name == 'widowx_bridge':
            self.obs_window.append(
                {
                    'qpos': obs['agent']['qpos'][:7], # TODO to modify as 8, using 2 fingers
                    'images':
                        {
                            # maniskill should be image instead of sensor data
                            cam_name: obs['sensor_data'][cam_name]['rgb'].permute(2, 0, 1).to(torch.uint8) if cam_name != 'background' else None
                                for cam_name in self.camera_names
                        }
                }
            )

    def reset(self, task_description: str) -> None:
        self.obs_window = None
        self.text_embedding = None
        self.last_instruction = None
        self.action_buffer = None
        self.internal_t = 0

    def step(
        self, observations: Dict, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            observations: accquired by 'env.get_obs()'
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            action: dict; policy action output
        """
        text_embedding = None
        if self.text_embedding_path is not None:
            text_embedding = torch.load(self.text_embedding_path)["embeddings"]
        qpos_action = self.predict_action(observations, task_description,text_embedding)
 
        if self.robot_name in ["widowx_bridge",'google_robot']:

            ee_pose = transfer_qpos_2_ee_pose(self.env, qpos_action[0][:6]) # just first 6 joints

            action = {}
            action["gripper"] = np.array([qpos_action[0][6]])
            action["world_vector"] = ee_pose.p.squeeze().numpy()
            # 重排列 quat
            action_euler = Rotation.from_quat(ee_pose.q.squeeze()[[1,2,3,0]]).as_euler('xyz', degrees=True)

            action_rotation_delta = np.asarray(action_euler, dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle * self.action_scale
            action["terminate_episode"] = np.array([0.0])

        elif self.robot_name in ["mobile_aloha","mobile_aloha_v2"]:
            action = qpos_action

        return qpos_action, action

    def visualize_epoch(
            self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
        ) -> None:
        # predicted_raw_actions = np.array(predicted_raw_actions)
        num_actions = predicted_raw_actions.shape[1]
        plt.figure(figsize=(5 * num_actions, 6))
        for i in range(num_actions):
            plt.subplot(2, num_actions//2, i + 1)
            if i // 7 ==0:
                plt.plot(predicted_raw_actions[:, i], label=f"Inference action Left Joint {(i+1) % 7}")
            else:
                plt.plot(predicted_raw_actions[:, i], label=f"Inference action Right Joint {(i+1) % 7}")
            plt.legend()
        plt.tight_layout()
        pic_name = "inference action_history"
        plt.savefig(os.path.join(save_path, pic_name+".png"))
        print("Save inference action_history.png")
        plt.close()


def transfer_qpos_2_ee_pose(
    env, 
    qpos : torch.Tensor,
):
    """Transfer joint positions to ee pose."""

    get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names)
    kinematics = Kinematics(
        urdf_path=env.agent.urdf_path,
        end_link_name=env.agent.ee_link_name,
        articulation=env.agent.robot,
        active_joint_indices=get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names),
    )

    ''' 
        google_robot

            robot.get_active_joints()
            ['joint_wheel_left', 'joint_wheel_right', 'joint_torso', 'joint_shoulder',
            'joint_bicep', 'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper',
            'joint_finger_right', 'joint_finger_left', 'joint_head_pan', 'joint_head_tilt']
            If robot is not mobile, then the first two joints are not active
            
            0 torso 
            1 shoulder
            2 bicep
            3 elbow
            4 forearm
            5 wrist
            6 gripper


        widowx_bridge

            0 waist
            1 shoulder
            2 elbow
            3 forearm_roll
            4 wrist_angle
            5 wrist_rotate

            6 left_finger
            7 right_finger

    '''
    qpos = qpos.squeeze()
    qpos = torch.from_numpy(qpos)
    qpos_fk = torch.zeros(qpos.shape[0], env.agent.robot.max_dof, # 8
                        dtype=qpos.dtype, device=env.agent.robot.device)
    qpos_fk[:, get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names)] = qpos
    ee_pose = kinematics.compute_fk(qpos_fk)
    ee_pose_tensor = ee_pose.raw_pose.squeeze(0)
    # print("transfer_qpos_2_ee_pose: ", ee_pose_tensor)

    return ee_pose