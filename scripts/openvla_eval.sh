# eval
ckpt_path="../openvla/checkpoints/grape_simpler_sft_dataset/steps_4000/merged_004000"
#ckpt_path="openvla/openvla-7b"
XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_visualize.py \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes 50 --num-envs 1 \
  --openvla_unnorm_key="grape_simpler_sft_dataset"