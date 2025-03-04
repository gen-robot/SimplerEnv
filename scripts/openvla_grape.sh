ckpt_path="ZijianZhang/OpenVLA-7B-SFT-Simpler"
XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_visualize.py \
  --model="openvla" --ckpt_path="${ckpt_path}" \
  -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes 50 --num-envs 1 \
  --openvla_unnorm_key="Simpler"