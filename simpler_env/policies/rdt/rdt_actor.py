import os
import yaml
import json
from collections import deque
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PImage

from third_party.rdt.constants import *
from third_party.rdt.models.multimodal_encoder.t5_encoder import T5Embedder
from third_party.rdt.configs import ROBOT_INDICES, ROBOT_CAMERA_NAMES
from simpler_env.policies.rdt.rdt_aloha_model import (
    create_model as create_RDT_model, 
    RoboticDiffusionTransformerModel
)

# TODO: currently, it does not support batched execution.

def dict_apply(d, func):
    """Applies `func` to all tensor leaves in a nested dictionary `d`."""
    if isinstance(d, dict):
        # Recursively apply to each element in the dictionary
        return {k: dict_apply(v, func) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        # Apply the function to the tensor leaf
        return func(d)
    else:
        # Return other types as-is
        return d

class RDTActor:
    def __init__(self, 
                 ctrl_freq: int=25, 
                 action_horizon: int=1,
                 camera_names: List[str]=None, 
                 model_cfg_path: str=None,
                 use_actions_interpolation: bool=False,
                 device: str='cuda',
                 dtype: torch.dtype=torch.bfloat16,
                 pretrained_checkpoint: str=None,
                 lora_adapter: str=None,
                 robot_name: str="mobile_aloha_v2",
                 *args,
                 **kwargs):
        self.ctrl_freq = ctrl_freq
        self.device = device
        self.dtype = dtype
        self.robot_name = robot_name

        self.use_actions_interpolation = use_actions_interpolation

        self.model_cfg_path = RDT_DEFAULT_CONFIG if model_cfg_path is None else model_cfg_path

        self.obs_window = None
        self.text_embedding = None

        # if camera_names is None:
        #     self.camera_names = deepcopy(ROBOT_CAMERA_NAMES[robot_name]) # to make sure we have 3 pictures(background) 
        if robot_name not in ROBOT_CAMERA_NAMES:
            raise ValueError(f"Unsupported robot name: {robot_name}")
        self.camera_names = deepcopy(ROBOT_CAMERA_NAMES[robot_name]) # to make sure we have 3 pictures(background)

        with open(self.model_cfg_path, "r") as fp:
            self.config = yaml.safe_load(fp)

        self.n_frames = self.config["common"]["img_history_size"]
        self.action_chunk_size = self.config["common"]["action_chunk_size"]
        self.action_horizon = min(action_horizon, self.action_chunk_size) if action_horizon > 0 else self.action_chunk_size

        # assert pretrained_checkpoint is not None, "Pretrained checkpoint must be provided."
        if pretrained_checkpoint is None or not os.path.exists(pretrained_checkpoint):
            pretrained_checkpoint = RDT1B_PATH
            print(f"[WARNING] Pretrained checkpoint not found. Using default checkpoint: {pretrained_checkpoint}")
        if lora_adapter is not None:
            assert os.path.exists(lora_adapter), f"LORA adapter not found at {lora_adapter}."
            # print(f"[INFO] Using LORA adapter: {lora_adapter}")

        self.pretrained_checkpoint = pretrained_checkpoint
        self.lora_adapter = lora_adapter

        self.make_policy()
        self.lang_tokenizer, self.lang_encoder = None, None

        self.last_instruction = None
        self.action_buffer = None
        self.internal_t = 0

        self.fk_pose = None

    def reset(self):
        self.obs_window = None
        self.text_embedding = None
        self.last_instruction = None
        self.action_buffer = None
        self.internal_t = 0

    def make_policy(self):
        model = create_RDT_model(
            args=self.config,
            device=self.device,
            dtype=self.dtype,
            pretrained=self.pretrained_checkpoint,
            lora_adapter=self.lora_adapter,
            pretrained_vision_encoder_name_or_path=SIGLIP_PATH,
            control_frequency=self.ctrl_freq,
            robot_name=self.robot_name
        )

        self.rdt_policy = model

        return self.rdt_policy
    
    def make_lang_models(self):
        # Note: if your GPU VRAM is less than 24GB, 
        # it is recommanded to enable offloading by specifying an offload directory.
        text_embedder = T5Embedder(
            from_pretrained=T5_PATH, 
            model_max_length=self.config["dataset"]["tokenizer_max_length"], 
            device=self.device,
            use_offload_folder=None # Specify your offload directory here, ensuring the directory exists.
        )

        self.lang_tokenizer, self.lang_encoder = text_embedder.tokenizer, text_embedder.model

        return self.lang_tokenizer, self.lang_encoder
    
    @torch.no_grad()
    def encode_instruction(self, instr: str):
        if self.lang_tokenizer is None or self.lang_encoder is None:
            self.make_lang_models()
        
        tokens = self.lang_tokenizer(
            instr, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(self.device)

        tokens = tokens.view(1, -1)
        
        pred = self.lang_encoder(tokens).last_hidden_state

        return pred
    
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

    # RDT inference
    @torch.no_grad()
    def infer(self):
        # fetch images in sequence [front, right, left]
        image_arrs = []
        for t in range(-self.n_frames, 0):  # n_frames = image_history_size
            for cam_name in self.camera_names:
                image_arrs.append(
                    self.obs_window[t]['images'][cam_name]
                )
        
        to_pil = transforms.ToPILImage()
        images = [to_pil(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # get last qpos in shape [N, ] and unsqueeze to [1, N], N should be 14 for agilex, but 7 for widowx
        proprio = self.obs_window[-1]['qpos']
        proprio = proprio.unsqueeze(0)
        print("---------------------")
        # print("observation proprio:",proprio)

        from simpler_env.policies.rdt.rdt_model import transfer_qpos_2_ee_pose
        from scipy.spatial.transform import Rotation as R
        if self.fk_pose == None:
            self.fk_pose = {}
        obs_ee_pose = transfer_qpos_2_ee_pose(self.env, proprio[:,:6])
        self.fk_pose["xyz"] = obs_ee_pose.p.squeeze().numpy()
        self.fk_pose["euler_angle"] = R.from_quat(obs_ee_pose.q.squeeze()[[1,2,3,0]]).as_euler('xyz', degrees=True)
        print("observation fk pose:", self.fk_pose)

        actions = self.rdt_policy.step(
            proprio=proprio,
            images=images,
            text_embeds=self.text_embedding
        )

        return actions

    def predict_action(self, obs: Dict, instr: str=None, text_embedding: torch.Tensor=None):
        if text_embedding is not None:
            print("[INFO] Using pre-computed language embedding.")
            self.text_embedding = text_embedding
        elif instr is not self.last_instruction:
            print("[INFO] Instruction changed. New instruction: ", instr)
            self.internal_t = 0
            self.last_instruction = instr
            self.text_embedding = self.encode_instruction(instr)

            # save_path = os.path.join(f"{instr}.pt")
            # torch.save({
            #         "instruction": instr,
            #         "embeddings": self.text_embedding
            #     }, save_path
            # )

        self.update_obs_window(
            dict_apply(obs, lambda x: torch.squeeze(x, dim=0)))

        if self.internal_t % self.action_horizon == 0:
            self.action_buffer = self.infer().cpu().numpy()

        raw_action = self.action_buffer[:, self.internal_t % self.action_horizon]
        self.internal_t += 1

        action = raw_action.copy()

        return action

    # def rdt_forward(self, batch, text_embedding=None):
    #     weight_dtype = self.dtype

    #     if text_embedding is not None:
    #         print("[INFO] Using pre-computed language embedding.")
    #         self.text_embedding = text_embedding

    #     if self.internal_t % self.action_horizon == 0:
    #         images = batch["images"].to(self.device, dtype=weight_dtype)
    #         states = batch["states"].to(self.device, dtype=weight_dtype) # (B, T, D_a)
    #         # We only use the last state as input
    #         states = states[:, -1:, :]
    #         actions = batch["actions"].to(self.device, dtype=weight_dtype)
    #         state_elem_mask = batch["state_elem_mask"].to(self.device, dtype=weight_dtype)
    #         ctrl_freqs = batch["ctrl_freqs"].to(self.device, dtype=weight_dtype)

    #         batch_size, _, C, H, W = images.shape
    #         image_embeds = self.rdt_policy.vision_model(images.reshape(-1, C, H, W)).detach()
    #         image_embeds = image_embeds.reshape((batch_size, -1, self.rdt_policy.vision_model.hidden_size))
    #         image_embeds = image_embeds.to(self.device)
            
    #         lang_attn_mask = batch["lang_attn_mask"].to(self.device)
    #         if self.text_embedding is None:
    #             text_embeds = self.lang_encoder(
    #                     input_ids=batch["input_ids"].to(self.device),
    #                     attention_mask=lang_attn_mask
    #                 )["last_hidden_state"].detach()
    #             self.text_embedding = text_embeds.to(self.device, dtype=weight_dtype)
                
    #         state_elem_mask = state_elem_mask.unsqueeze(1)

    #         trajectory = self.rdt_policy.predict_action(
    #             lang_tokens=self.text_embedding,
    #             lang_attn_mask=lang_attn_mask,
    #             img_tokens=image_embeds,
    #             state_tokens=states,
    #             action_mask=state_elem_mask,
    #             ctrl_freqs=ctrl_freqs
    #         )

    #         self.action_buffer = trajectory.detach().cpu().numpy()

    #     raw_action = self.action_buffer[:, self.internal_t % self.action_horizon]
    #     self.internal_t += 1

    #     action = raw_action.copy()
        
    #     return action
    