ckpt_path="/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/third_party/openvla/checkpoints/panda_simpler_sft_new_dataset/steps_8000/merged_008000"
unnorm_key="panda_simpler_sft_new_dataset"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m simpler_env.scripts.openvla_replay \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1" -s 0 --num-episodes 100 --num-envs 10 --save-video \
  --openvla_unnorm_key="${unnorm_key}" --policy_setup panda --max_episode_len 100