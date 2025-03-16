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

### openvla infer

```bash
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

### openvla train

```bash
git clone git@github.com:gen-robot/openvla.git && cd openvla && git checkout dev-jijia && pip install -e . && cd ..
#git clone https://github.com/openvla/openvla.git && cd openvla && pip install -e . && cd ..
pip install -U tyro
pip intall datasets==3.3.2

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### grape

```bash
git clone https://github.com/DelinQu/SimplerEnv-OpenVLA --recurse-submodules

git clone https://github.com/aiming-lab/GRAPE.git

git clone https://github.com/kpertsch/rlds_dataset_builder.git
cd rlds_dataset_builder
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```

### octo

```bash
git clone https://github.com/octo-models/octo.git
cd octo && pip install -e .
pip install -r requirements.txt
pip install -U tryo
pip install numpy==1.24.4
pip install "jax[cuda12_pip]==0.4.20" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

huggingface-cli download rail-berkeley/octo-base
```

## run

### openvla maniskill3

```bash
# PutSpoonOnTableClothInScene-v1
# PutCarrotOnPlateInScene-v1
# StackGreenCubeOnYellowCubeBakedTexInScene-v1
# PutEggplantInBasketScene-v1


# grape
#ckpt_path="openvla/openvla-7b"
#unnorm_key="bridge_orig"

ckpt_path="ZijianZhang/OpenVLA-7B-SFT-Simpler"
#ckpt_path="../openvla/results/grape/adapter/openvla-7b+grape_simpler_dpos_dataset+b1+lr-2e-05+lora-r32+dropout-0.0/d1121_check_merged"
unnorm_key="Simpler"
#unnorm_key="bridge_orig"

#ckpt_path="../openvla/checkpoints/grape_simpler_sft_dataset_268/steps_4000/merged_004000"
#unnorm_key="grape_simpler_sft_dataset_268"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect.py \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes 50 --num-envs 50 --save-video \
  --openvla_unnorm_key="${unnorm_key}"

# dpo
# "PutSpoonOnTableClothInScene-v1" 
ckpt_path="ZijianZhang/OpenVLA-7B-SFT-Simpler"
for task in "StackGreenCubeOnYellowCubeBakedTexInScene-v1" "PutEggplantInBasketScene-v1"; do
  CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect_dpo.py \
    --model="openvla" --ckpt_path="${ckpt_path}" -e "${task}" --num-envs 4 \
    --openvla_unnorm_key="Simpler"
done
  
# ppo
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/train_ms3_ppo.py

flameprof --format=svg --threshold=0.1 images/perf/perf.bin > images/perf/perf.svg
```

### openvla maniskill2

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
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;
  
# action_ensemble_temp="-0.8"
# --action-ensemble-temp ${action_ensemble_temp}
```

### grape

```bash
python simpler_env/main_inference.py --policy-model openvla --ckpt-path "openvla/openvla-7b" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
```

### octo

```bash
# 42 %
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect.py \
--model="octo-small" -e PutSpoonOnTableClothInScene-v1 -s 0 --num-episodes 256 --num-envs 64

# 14 %
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect.py \
--model="octo-small" -e PutCarrotOnPlateInScene-v1 -s 0 --num-episodes 512 --num-envs 64

# 2 %
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect.py \
--model="octo-small" -e StackGreenCubeOnYellowCubeBakedTexInScene-v1 -s 0 --num-episodes 2880 --num-envs 144

# 57 %
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect.py \
--model="octo-small" -e PutEggplantInBasketScene-v1 -s 0 --num-episodes 256 --num-envs 64

CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect.py \
--model="octo-small" -e PandaPutSpoonOnTableClothInScene-v1 -s 0 --num-episodes 256 --num-envs 64 \
--policy_setup="panda" --save_video --save_data

# PutSpoonOnTableClothInScene-v1
# PutCarrotOnPlateInScene-v1
# StackGreenCubeOnYellowCubeBakedTexInScene-v1
# PutEggplantInBasketScene-v1

tfds build --overwrite
```

## train

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes=1 --nproc-per-node 1 vla-scripts/finetune_grape.py \
  --vla_path "openvla/openvla-7b" \
  --dataset_s_name "grape_simpler_dpos_dataset" \
  --dataset_f_name "grape_simpler_dpof_dataset" \
  --traj_dir "../datasets" \
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


## panda

### collect data via motion planning

```bash
## save .npy file
# simpler stack cube
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1 \
--only_count_success --save_video --save_data --control_mode pd_ee_target_delta_pose --num_traj 5 --num_procs 1

# simpler put spoon
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaPutSpoonOnTableClothInScene-v1 \
--only_count_success --save_video --save_data --control_mode pd_ee_target_delta_pose --num_traj 5 --num_procs 1

# simpler put carrot
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaPutCarrotOnPlateInScene-v1 \
--only_count_success --save_video --save_data --control_mode pd_ee_target_delta_pose --num_traj 5 --num_procs 1 

# If you want local visulization, you should add "--vis", but if you add both "--vis" and "--save_video",
# the video saved might have some error patch in the picture.
# Now we not support "--sim_backend gpu", for the error in motion planning, which is caused by _initialize_episode
# function in panda env.

# In local computer
rsync -avzP scp/ wq3:/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/videos/scp

## save .h5 file
# stack cube
# python -m mani_skill.examples.motionplanning.panda.run_simpler -e PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1 \
# --only_count_success --traj_name "bingwen" --save_video --num_traj 5 --num_procs 1

# # put spoon
# python -m mani_skill.examples.motionplanning.panda.run_simpler -e PandaPutSpoonOnTableClothInScene-v1 \
# --only_count_success --traj_name "bingwen" --save_video --num_traj 5 --num_procs 1

# # put carrot
# python -m mani_skill.examples.motionplanning.panda.run_simpler -e PandaPutCarrotOnPlateInScene-v1 \
# --only_count_success --traj_name "bingwen" --save_video --num_traj 5 --num_procs 1

# support -> PandaStackGreenCubeOnYellowCubeBakedTexInScene
# support -> PandaPutSpoonOnTableClothInScene
# support -> PandaPutCarrotOnPlateInScene
# not support -> PandaPutEggplantInBasketScene
```


### openvla
#### evaluate
```bash
# ckpt_path="openvla/openvla-7b"
# ckpt_path="ZijianZhang/OpenVLA-7B-SFT-Simpler"
# unnorm_key="Simpler"

# stack cube
ckpt_path="/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/third_party/openvla/checkpoints/panda_simpler_sft_dataset/steps_10000/merged_010000"
unnorm_key="panda_simpler_sft_dataset"
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m simpler_env.eval_ms3_collect \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1" -s 0 --num-episodes 100 --num-envs 10 --save-video \
  --openvla_unnorm_key="${unnorm_key}" --policy_setup panda --max_episode_len 100

# put carrot
ckpt_path="/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/third_party/openvla/checkpoints/panda_simpler_sft_dataset/steps_10000/merged_010000"
unnorm_key="panda_simpler_sft_dataset"
CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m simpler_env.eval_ms3_collect \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PandaPutCarrotOnPlateInScene-v1" -s 0 --num-episodes 100 --num-envs 10 --save-video \
  --openvla_unnorm_key="${unnorm_key}" --policy_setup panda --max_episode_len 100

# put spoon
ckpt_path="/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/third_party/openvla/checkpoints/panda_simpler_sft_dataset/steps_10000/merged_010000"
unnorm_key="panda_simpler_sft_dataset"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m simpler_env.eval_ms3_collect \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PandaPutSpoonOnTableClothInScene-v1" -s 0 --num-episodes 100 --num-envs 10 --save-video \
  --openvla_unnorm_key="${unnorm_key}" --policy_setup panda --max_episode_len 100
```

### rdt
```bash
# not test

# put carrot
python -m simpler_env.eval_ms3_collect --model rdt --ckpt_path '/nvme_data/embodied_agent/pretrained/rdt-1b' \
--env_id "PutCarrotOnPlateInScene-v1" --policy_setup widwox_bridge

# put eggplant
python -m simpler_env.eval_ms3_collect --model rdt --ckpt_path '/nvme_data/embodied_agent/pretrained/rdt-1b' \
--env_id "PutEggplantInBasketScene-v1" --policy_setup widowx_bridge

# put spoon
python -m simpler_env.eval_ms3_collect --model rdt --ckpt_path '/nvme_data/embodied_agent/pretrained/rdt-1b' \
--env_id "PutSpoonOnTableClothInScene-v1" --policy_setup widowx_bridge

# stack cube
python -m simpler_env.eval_ms3_collect --model rdt --ckpt_path '/nvme_data/embodied_agent/pretrained/rdt-1b' \
--env_id "StackGreenCubeOnYellowCubeBakedTexInScene-v1" --policy_setup widowx_bridge
```