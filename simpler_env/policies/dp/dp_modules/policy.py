import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from simpler_env.policies.dp.dp_modules.models.base_nets import ResNet18Conv, SpatialSoftmax
from simpler_env.policies.dp.dp_modules.models.diffusion_nets import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.num_images = args_override.get('num_images', 1)

        self.observation_horizon = args_override.get('observation_horizon', 1)
        # self.action_horizon = args_override.get('action_horizon', 1)          # apply chunk size
        self.prediction_horizon = args_override.get('prediction_horizon', 1)    # chunk size
        self.num_inference_timesteps = args_override.get('num_inference_timesteps', 10)
        self.ema_power = args_override.get('ema_power', 0.75)
        self.lr = args_override.get('lr', 1e-5)
        self.weight_decay = args_override.get('weight_decay', 0)

        self.num_kp = 32
        self.feature_dimension = args_override.get('feature_dim', 64)
        self.qpos_dim = args_override.get('global_obs_dim', 10)
        self.ac_dim = args_override.get('action_dim', 10)
        self.obs_dim = self.feature_dimension * self.num_images + self.qpos_dim  # camera features and proprio

        self.resnet_pretrained = args_override.get("resnet_pretrained", True)

        backbones = []
        pools = []
        linears = []
        for _ in range(self.num_images):
            backbones.append(
                ResNet18Conv(
                    **{
                        'input_channel': 3,
                        'pretrained': self.resnet_pretrained,
                        'input_coord_conv': False
                    }
                )
            )
            pools.append(
                SpatialSoftmax(
                    **{
                        # 'input_shape': [512, 8, 10], # for (240, 320)
                        'input_shape': [512, 7, 7],  # for (224, 224)
                        'num_kp': self.num_kp,
                        'temperature': 1.0,
                        'learnable_temperature': False,
                        'noise_std': 0.0
                    }
                )
            )
            linears.append(
                torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension)
            )

        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        backbones = replace_bn_with_gn(backbones)  # TODO

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon,
            input_len=self.prediction_horizon,
            diffusion_step_embed_dim=args_override.get("diffusion_step_embed_dim", 256)
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(self.num_images):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                # print(cam_features.shape)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps).float()

            # predict the noise residual
            # print(noisy_actions.shape)
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            # L1 loss
            all_l1 = F.l1_loss(noise_pred, noise, reduction='none')
            l2_loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()
            l1_loss = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = l2_loss
            loss_dict['l1_loss'] = l1_loss
            loss_dict['loss'] = l1_loss + l2_loss
            # loss_dict['all_l2'] = all_l2  ###

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else:  # inference time
            # To = self.observation_horizon
            # Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model

            all_features = []
            for cam_id in range(self.num_images):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # print(naction.shape, obs_cond.shape)
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # print(noise_pred.shape)

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status
