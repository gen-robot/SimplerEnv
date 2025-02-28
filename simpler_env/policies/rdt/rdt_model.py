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

from third_party.rdt.data.utils import *
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
        self.env = kwargs["env"]
        self.robot_name = kwargs["robot_name"]
        self.action_scale = kwargs["action_scale"]
        self.text_embedding_path = text_embedding_path
        self.last_action = None
        super().__init__(*args, **kwargs)

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

        # two arms in maniskill3
        if self.robot_name in ['mobile_aloha','mobile_aloha_v2','rdt']:
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
        elif self.robot_name in ['widowx_bridge']:
            ee_pose = self.transfer_qpos_2_ee_pose(obs['agent']['qpos'][...,:6], world_frame=False) # root frame
            ee_pose_xyz = ee_pose.p.squeeze()
            ee_rot_matrix = R.from_quat(ee_pose.q.squeeze()[[1,2,3,0]]).as_matrix()[np.newaxis,:,:]
            ee_pose_rot_angle = torch.from_numpy(
                                        rotation_matrix_to_ortho6d(tf.convert_to_tensor(ee_rot_matrix)).numpy()
                                    )
            self.obs_window.append(
                {
                    # arm_joint_pos:6, gripper_joint_0_pos:1, eef_pos:3,eef_angle:6 -> 16 dimensional
                    'qpos': obs['agent']['qpos'][:7],
                    'eef_pos_rot6d':  torch.cat([ee_pose_xyz, ee_pose_rot_angle[0,:6]]).to(torch.float32),
                    'qpos_vel': obs['agent']['qvel'][:7], # controlled by enable_qvel_obs
                    'images':
                        {
                            # maniskill should be image instead of sensor data
                            cam_name: obs['sensor_data'][cam_name]['rgb'].permute(2, 0, 1).to(torch.uint8) if cam_name != 'background' else None
                                for cam_name in self.camera_names
                        },
                }
            )
            # obs['sensor_data'][cam_name]['rgb'] -> (480,640,3) -> [H,W,C]  ->(permute) [C,H,W]
        else:
            ValueError(f"robot_name:{self.robot_name} error in update_obs_window()")

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
                gripper
                world_vector
                rot_axangle
        """
        text_embedding = None
        if self.text_embedding_path is not None:
            text_embedding = torch.load(self.text_embedding_path)["embeddings"]
        qpos_action = self.predict_action(observations, task_description, text_embedding)
 
        if self.robot_name in ["widowx_bridge"]:
            raw_action = qpos_action
            delta = False
            joint_fk = True
            action = {}
            if delta:
                last_obs_state = {}
                last_obs_state['qpos'] = self.obs_window[-1]["qpos"].numpy()
                last_obs_state['eef_pos_rot6d'] = self.obs_window[-1]["eef_pos_rot6d"].numpy()
                if joint_fk:
                    ee_pose = self.transfer_qpos_2_ee_pose(qpos_action[0,:6], world_frame=False)
                    action["world_vector"] = (ee_pose.p.squeeze().numpy() - last_obs_state['eef_pos_rot6d'][:3]) * self.action_scale
                    matrix_new = R.from_quat(ee_pose.q.squeeze()[[1,2,3,0]])
                    matrix_old = R.from_matrix(ortho6d_to_rotation_matrix(
                        tf.convert_to_tensor(last_obs_state['eef_pos_rot6d'][3:], dtype=tf.float32)
                        ).numpy())
                    action['rot_axangle'] = (matrix_new * matrix_old.inv()).as_euler('xyz', degrees = False) * self.action_scale # radian 
                    action['gripper'] = qpos_action[0,6:7]
                    action["terminate_episode"] = np.array([0.0])
                else:
                    matrix_new = R.from_matrix(ortho6d_to_rotation_matrix(
                                            tf.convert_to_tensor(qpos_action[0,10:16], dtype=tf.float32)
                                            ).numpy())
                    matrix_old = R.from_matrix(ortho6d_to_rotation_matrix(
                                            tf.convert_to_tensor(last_obs_state['eef_pos_rot6d'][3:], dtype=tf.float32)
                                            ).numpy())
                    action['world_vector'] = (qpos_action[0,7:10] - last_obs_state['eef_pos_rot6d'][:3])* self.action_scale
                    action['rot_axangle'] = (matrix_new * matrix_old.inv()).as_euler('xyz', degrees = False) * self.action_scale # radian 
                    action['gripper'] = qpos_action[0,6:7]
                    action["terminate_episode"] = np.array([0.0])
            else:
                self.env.unwrapped.agent.controller.controllers['arm'].config.use_delta = False
                self.env.unwrapped.agent.controller.controllers['arm'].config.frame = 'root_translation:root_aligned_body_rotation'
                if joint_fk:
                    ee_pose = self.transfer_qpos_2_ee_pose(qpos_action[0,:6], world_frame=False) # 在 arm_root 坐标系下
                    action["world_vector"] = ee_pose.p.squeeze().numpy() * self.action_scale
                    matrix = R.from_quat(ee_pose.q.squeeze()[[1,2,3,0]])
                    action['rot_axangle'] = matrix.as_euler('xyz', degrees = False) * self.action_scale # radian 
                    action['gripper'] = qpos_action[0,6:7]
                    action["terminate_episode"] = np.array([0.0])
                else:
                    rotation_matrix = ortho6d_to_rotation_matrix(
                                            tf.convert_to_tensor(qpos_action[0,10:16], dtype=tf.float32)
                                            ).numpy()
                    action['world_vector'] = qpos_action[0,7:10] * self.action_scale
                    action['gripper'] = qpos_action[0,6:7]
                    action['rot_axangle'] = R.from_matrix(rotation_matrix).as_euler('xyz', degrees = False) * self.action_scale # radian 
                    action["terminate_episode"] = np.array([0.0])

        elif self.robot_name in ["mobile_aloha","mobile_aloha_v2",'rdt']:
            raw_action = qpos_action
            action = raw_action
        else:
            raise ValueError("robot_name error")
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


    # RDT inference
    @torch.no_grad()
    def infer(self, unnorm_output=True):
        # fetch images in sequence [front, right, left]
        image_arrs = []
        for t in range(-self.n_frames, 0):  # n_frames = image_history_size
            for cam_name in self.camera_names:
                image_arrs.append(
                    self.obs_window[t]['images'][cam_name]
                )
        
        # image_arrs[3].shape -> [3,480,640] -> [C,H,W]
        to_pil = transforms.ToPILImage()
        images = [to_pil(arr) if arr is not None else None
                  for arr in image_arrs]
        # images -> [640, 480] -> [W,H]
        if self.robot_name in ['mobile_aloha','mobile_aloha_v2','rdt']:
            # get last qpos in shape [14, ] and unsqueeze to [1, 14]
            proprio = self.obs_window[-1]['qpos']
            proprio = proprio.unsqueeze(0)

            actions = self.rdt_policy.step(
                proprio=proprio,
                images=images,
                text_embeds=self.text_embedding,
                unnorm_output=unnorm_output
            )

        elif self.robot_name in ['widowx_bridge']:
            proprio = None
            eef_pos_rot6d = None
            proprio_qvel = None
            # get last qpos in shape [14, ] and unsqueeze to [1, 14]
            proprio = self.obs_window[-1]['qpos'].unsqueeze(0)
            eef_pos_rot6d = self.obs_window[-1]['eef_pos_rot6d'].unsqueeze(0)
            proprio_qvel = self.obs_window[-1]['qpos_vel'].unsqueeze(0)
            print("---------------------")
            print("observation proprio:",proprio)
            print("observation eef_pos_rot6d:",eef_pos_rot6d)
            print("observation proprio_qvel:",proprio_qvel)
            actions = self.rdt_policy.step(
                proprio=proprio,
                images=images,
                text_embeds=self.text_embedding,
                unnorm_output=unnorm_output,
                eef_pos_rot6d = eef_pos_rot6d,
                proprio_qvel = proprio_qvel
            )
        else:
            raise ValueError("error robot_name in infer()")

        return actions


    def transfer_qpos_2_ee_pose(self, qpos, world_frame:bool = False):
        """Transfer joint positions to ee pose in the world frame"""
        env = self.env
        # get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names) -> [0,1,2,3,4,5]
        kinematics = Kinematics(
            urdf_path=env.agent.urdf_path,
            end_link_name=env.agent.ee_link_name,
            articulation=env.agent.robot,
            active_joint_indices=get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names),
        )
        ''' 
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

        if world_frame: 
            return ee_pose * env.agent.robot.root.pose  # 即为 env.agent.robot.find_link_by_name(env.agent.ee_link_name).pose
        else:
            return ee_pose # in robot_root_frame
