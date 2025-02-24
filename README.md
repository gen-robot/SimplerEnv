# Simpler

## install

```bash
conda create -n simpler -y python=3.10
conda activate simpler
mkdir thirdparty && cd thirdparty
```

### simpler (maniskill3)

```bash
#git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules && cd SimplerEnv && git checkout maniskill3 && cd ..
pip install --upgrade git+https://github.com/haosulab/ManiSkill.git
cd SimplerEnv && pip install -e . && cd ..
```

### simpler (maniskill2)

```bash
cd SimplerEnv
pip install -e .
cd ManiSkill2_real2sim && pip install -e . && cd ..
conda install -c conda-forge -y ffmpeg=4.2.2
pip install matplotlib mediapy "gymnasium>=0.28.1,<1.0" numpy==1.24.4
#pip install tensorflow[and_cuda]
```

### openvla

```bash
git clone https://github.com/openvla/openvla.git && cd openvla && pip install -e . && cd ..

#pip install -r requirements_full_install.txt
pip install matplotlib mediapy
pip install dm-tree
pip install transformers==4.40.1 torchvision timm==0.9.10 tokenizers==0.19.1 accelerate

pip install datasets
pip install ninja
pip install flash-attn --no-build-isolation
#pip install flash-attn==2.6.1 --no-build-isolation
#pip install flash-attn==2.5.5 --no-build-isolation
```

## run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/real2sim_eval_maniskill3.py \
  --model="openvla" --ckpt_path="openvla/openvla-7b" \
  -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes 20 --num-envs 1
  
# PutCarrotOnPlateInScene-v1
# PutEggplantInBasketScene-v1
```

```bash
ckpt_path="openvla/openvla-7b" # "/home/jijia/nfs/Project/RLVLA/thirdparty/models/openvla-7b"
policy_model="openvla"

logging_dir="results/openvla-7b${action_ensemble_temp}"
gpu_id=1


scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --logging-dir ${logging_dir} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 20 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;
  
# action_ensemble_temp="-0.8"
# --action-ensemble-temp ${action_ensemble_temp}
```

```bash
torchrun --standalone --nnodes=1 --nproc-per-node 2 simpler_env/runner/finetune_grape.py \
  --vla_path "openvla/openvla-7b" \
  --dataset_name "put carrot on plate" \
  --chosen_traj_dir "results" \
  --rejected_traj_dir "results" \
  --run_root_dir "results/grape/root" \
  --adapter_tmp_dir "results/grape/adapter" \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project "rlvla" \
  --wandb_entity "hosnls" \
  --save_steps 1000
```

