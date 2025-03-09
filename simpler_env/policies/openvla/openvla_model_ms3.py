from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
from typing import List
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
        elif "panda" in policy_setup or "franka" in policy_setup: # TODO
            if "ZijianZhang" in saved_model_path:
                unnorm_key = "Simpler" if unnorm_key is None else unnorm_key
            else:
                unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
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

    # TODO with the version of jijia
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (B, H, W, 3), uint8, where B is the batch size
            task_description: Optional[str], list of task descriptions; if different from previous task descriptions, policy state is reset
        Output:
            raw_actions: dict; raw policy action output for all scenes
            actions: dict; processed actions to be sent to the maniskill2(axangle)/maniskill3(eulerangle) environment, with the following keys:
                - 'world_vector': np.ndarray of shape (B, 3), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (B, 3), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (B, 1), gripper action
                - 'terminate_episode': np.ndarray of shape (B, 1), 1 if episode should be terminated, 0 otherwise
        """
        # Ensure input is a batch
        image = image.cpu().numpy()
        assert len(image.shape) == 4, "Input image must be a batch of shape (B, H, W, 3)"
        batch_size = image.shape[0]
        assert isinstance(task_description, list)
        task_description = task_description[0]

        # Reset policy state if task_description changes
        if task_description is not None: # now only support the same description for all enviroments
            if task_description != self.task_description:
                self.reset(task_description)

        # Process images and task descriptions
        raw_actions = []
        actions = []

        for i in range(batch_size):
            # Process image
            img = image[i]
            assert len(img.shape) == 3 and img.dtype == np.uint8, "Each image must be of shape (H, W, 3) and dtype uint8"
            img_post = self._resize_image(img)
            img_post = Image.fromarray(img_post)

            # Process task description
            prompt = task_description

            # Predict action (7-dof; un-normalize for bridgev2)
            inputs = self.processor(prompt, img_post).to("cuda:0", dtype=torch.bfloat16)
            raw_action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=True)[None]
            # print(f"raw action: {raw_actions.shape}") # (1, 7)

            raw_action = {
                "world_vector": np.array(raw_action[0, :3]),
                "rotation_delta": np.array(raw_action[0, 3:6]),
                "open_gripper": np.array(raw_action[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
            }
            # Store raw action
            raw_actions.append(raw_action)
            # Process raw_action to obtain the action to be sent to the maniskill2 / maniskill3 environment
            action = {}
            action["world_vector"] = raw_action["world_vector"] * self.action_scale
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle # action_rotation_delta * self.action_scale

            # Gripper action logic. Note that the sticky gripper logic is only supported for the google_robot policy setup.
            # sticky_gripper_num_repeat not support for more than one simulation environment; otherwise, it will cause chaos.
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
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            elif self.policy_setup == "panda":
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

            action["terminate_episode"] = np.array([0.0])

            # Store processed action
            actions.append(action)

        # Convert lists to batched outputs
        raw_actions = {k: torch.stack([torch.from_numpy(ra[k]) for ra in raw_actions]) for k in raw_actions[0].keys()}
        actions = {k: torch.stack([torch.from_numpy(a[k]) for a in actions]) for k in actions[0].keys()}

        return raw_actions, actions

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        # print(image.shape) # (480, 640, 3)
        # print(self.image_size) # [224, 224]

        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

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
