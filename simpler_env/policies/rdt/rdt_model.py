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
from scipy.spatial.transform import Rotation as R

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
        elif self.robot_name in ['widowx_bridge', 'google_robot']:
            self.obs_window.append(
                {
                    'qpos': obs['agent']['qpos'],
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

    def _get_relative_rotation_axis_angle(self, euler_old, euler_new, seq='xyz'):
        """
        计算从 old 坐标系 旋转到 new 坐标系 的相对旋转，并转换为轴角表示。

        参数:
        euler_old: (3,) numpy 数组，旧坐标系相对于基坐标系的欧拉角 (单位: 弧度)
        euler_new: (3,) numpy 数组，新坐标系相对于基坐标系的欧拉角 (单位: 弧度)
        seq: 欧拉角顺序, 默认为 'xyz'

        返回:
        axis: (3,) numpy 数组, 旋转轴
        angle: float, 旋转角度 (单位: 弧度)
        """

        # 计算旧坐标系和新坐标系相对于基坐标系的旋转矩阵
        R_old = R.from_euler(seq, euler_old)  # 旧坐标系的旋转矩阵
        R_new = R.from_euler(seq, euler_new)  # 新坐标系的旋转矩阵

        # 计算相对旋转矩阵 R_rel = R_new * R_old^-1
        R_rel = R_new * R_old.inv()  # 计算 old → new 的旋转

        # 转换为轴角表示
        rotvec = R_rel.as_rotvec()  # 旋转向量 (旋转轴 * 旋转角度)
        angle = np.linalg.norm(rotvec)  # 旋转角度
        axis = rotvec / angle if angle > 1e-6 else np.array([1, 0, 0])  # 归一化旋转轴

        return axis, angle


    def step(
        self, observations: Dict, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            observations: accquired by 'env.get_obs()'
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            action: dict; policy action output
                gripper
                world_vector
                rot_axangle
        """
        text_embedding = None
        if self.text_embedding_path is not None:
            text_embedding = torch.load(self.text_embedding_path)["embeddings"]
        qpos_action = self.predict_action(observations, task_description, text_embedding)
 
        if self.robot_name in ["widowx_bridge",'google_robot']:
            ee_pose = transfer_qpos_2_ee_pose(self.env, qpos_action[0][:6]) # just first 6 joints

            raw_action = {}
            raw_action["world_vector"] = ee_pose.p.squeeze().numpy()
            raw_action["euler_angle"] = R.from_quat(ee_pose.q.squeeze()[[1,2,3,0]]).as_euler('xyz', degrees=True) 

            action = {}
            action["world_vector"] = (raw_action["world_vector"] - self.fk_pose["xyz"]) * self.action_scale

            # TODO to update
            action["gripper"] = torch.tensor([(2*((qpos_action[0][6]+qpos_action[0][7])>1)-1)])*self.action_scale
            action_rotation_ax, action_rotation_angle = self._get_relative_rotation_axis_angle(
                                                    np.radians(self.fk_pose["euler_angle"]),
                                                    np.radians(raw_action["euler_angle"])
                                                    )
            # print("rotation_ax", action_rotation_ax)
            # print("rotation_angle", action_rotation_angle)
            action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale
            action["terminate_episode"] = np.array([0.0])

        elif self.robot_name in ["mobile_aloha","mobile_aloha_v2"]:
            raw_action = qpos_action
            action = raw_action

        return raw_action, action

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
    qpos,
):
    """Transfer joint positions to ee pose."""

    # get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names) -> [0,1,2,3,4,5]
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
    qpos = torch.as_tensor(qpos)
    qpos_fk = torch.zeros(qpos.shape[0], env.agent.robot.max_dof, # 8
                        dtype=qpos.dtype, device=env.agent.robot.device)
    qpos_fk[:, get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names)] = qpos
    ee_pose = kinematics.compute_fk(qpos_fk)
    ee_pose_tensor = ee_pose.raw_pose.squeeze(0)
    # print("transfer_qpos_2_ee_pose: ", ee_pose_tensor)

    return ee_pose