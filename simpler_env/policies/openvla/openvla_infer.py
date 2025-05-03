from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from transformers import AutoConfig, AutoImageProcessor
import torch

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


class OpenVLAInference:
    def __init__(
        self,
        saved_model_path: str = "openvla/openvla-7b",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        elif "panda" in policy_setup:
            if "ZijianZhang" in saved_model_path:
                unnorm_key = "Simpler" if unnorm_key is None else unnorm_key
            else:
                unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        else:
            raise NotImplementedError(f"see huggingface config.json file.")
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        image_processor = PrismaticImageProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("openvla/openvla-7b",
                                                  trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            "openvla/openvla-7b",
            image_processor=image_processor,
            tokenizer=tokenizer,
            trust_remote_code=True
        )
        self.vla = OpenVLAForActionPrediction.from_pretrained(
            saved_model_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, images: torch.Tensor, task_description: list[str]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Input:
            image: torch.Tensor of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)
        task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: " for t in task_description]

        batch_size = images.shape[0]
        images = images.permute(0, 3, 1, 2) # [B, C, H, W]
        images = images.to("cuda:0", dtype=torch.bfloat16)

        inputs = self.processor(task_prompt, images).to("cuda:0", dtype=torch.bfloat16)

        # input_ids: torch.Size([1, 6])
        # attention_mask: torch.Size([1, 6])
        # pixel_values: torch.Size([1, 6, 224, 224])

        raw_actions = self.vla.predict_action_batch(**inputs, unnorm_key=self.unnorm_key, do_sample=True)

        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            "open_gripper": np.array(raw_actions[:, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale # [B, 3]

        # action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64) # [B, 3]
        # act_rotation = [euler2axangle(a[0], a[1], a[2]) for a in action_rotation_delta] # [B, 2]
        # rax = np.array([a[0] for a in act_rotation]) # [B, 3]
        # rag = np.array([a[1] for a in act_rotation]) # [B]
        # axangle = rax * rag.reshape(-1, 1) # [B, 3]
        action["rot_axangle"] = raw_action["rotation_delta"] * self.action_scale # [B, 3]

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0 # [B, 1]
        elif self.policy_setup == "panda":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0 # [B, 1]

        action["terminate_episode"] = np.array([0.0] * batch_size).reshape(-1, 1) # [B, 1]

        raw_action = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_action.items()}
        action = {k: torch.tensor(v, dtype=torch.float32) for k, v in action.items()}

        return raw_action, action

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
