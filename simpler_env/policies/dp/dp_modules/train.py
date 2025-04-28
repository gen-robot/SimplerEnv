import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import wandb
import time
from torchvision import transforms

# from datasets.dataset import load_sim2sim_data   #diffusion_policy.
from datasets.dataset_array import load_sim2sim_data   #diffusion_policy.

from utils.utils import compute_dict_mean, set_seed  #diffusion_policy.
from policy import DiffusionPolicy  #diffusion_policy.

import datetime

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stamp = datetime.datetime.now().strftime("%m%d_%H%M")

def main(args):
    set_seed(0)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    batch_size = args['batch_size']
    num_steps = args['num_steps']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    chunk_size = args['chunk_size']

    if args["wandb_offline"]:
        os.environ["WANDB_MODE"] = "offline"

    data_roots = [args['data_root'] ]

    name = f"cano_4_{stamp}" # change here
    num_seeds = 5000 # 

    camera_names = ["third"]
    usages = ["obs"]

    data_config = {
        "data_roots": data_roots,
        "num_seeds": num_seeds, 
        "camera_names": camera_names,
        "usages": usages,
        "chunk_size": chunk_size,
        "train_batch_size": batch_size,
        "val_batch_size": batch_size,
    }

    policy_config = {
        'lr': args['lr'],
        'num_images': len(camera_names) * len(usages),
        'action_dim': 10,
        'observation_horizon': 1,
        'action_horizon': 1,
        'prediction_horizon': args['chunk_size'],
        'global_obs_dim': 10,
        'num_inference_timesteps': 10,
        'ema_power': 0.75,
        'vq': False,
    }

    config = {
        'num_steps': num_steps,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'lr': args['lr'],
        'policy_config': policy_config,
        'data_config': data_config,
        'seed': args['seed'],
        'batch_size': batch_size,
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]

    train_dataloader, val_dataloader, train_norm_stats, val_norm_stats = load_sim2sim_data(
        **data_config
    )

    if not is_eval:
        wandb.init(project="diffusion_policy", reinit=True, name=name)
        wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    wandb.finish()


def make_policy(policy_config):
    policy = DiffusionPolicy(policy_config)
    return policy


def make_optimizer(policy):
    optimizer = policy.configure_optimizers()

    return optimizer


def forward_pass(data, policy):
    images = data["images"]
    qpos = data["proprio_state"]
    action = data["action"]
    is_pad = data["is_pad"]
    # image_data, qpos_data, action_data, is_pad = data
    images, qpos, action, is_pad = images.cuda(), qpos.cuda(), action.cuda(), is_pad.cuda()
    return policy(qpos, images, action, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_config)

    if config['resume_ckpt_path'] is not None:
        if os.path.exists(config['resume_ckpt_path']):
            loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
            print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
        else:
            raise Exception(f'Checkpoint at {config["resume_ckpt_path"]} not found!')

    policy.cuda()
    optimizer = make_optimizer(policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps + 1)):
        # validation
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)
            wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step)  # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    # ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    # torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir',
                        default="/home/zhouzhiting/Data/panda_data/diffusion_policy_checkpoints/" + now)
    parser.add_argument('--data_root', action='store', type=str, help='data_root', 
                        default="/home/zhouzhiting/Data/panda_data/cano_policy_pd_3")
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--validate_every', action='store', type=int, default=2000, help='validate_every',
                        required=False)
    parser.add_argument('--save_every', action='store', type=int, default=20000, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)

    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--wandb_offline', action='store_true', help='wandb_offline')
    main(vars(parser.parse_args()))

    # in docker
    # CUDA_VISIBLE_DEVICES=0 python3 -m diffusion_policy.train --chunk_size 20 --batch_size 128 --num_steps 500000  --lr 1e-5 --seed 0 