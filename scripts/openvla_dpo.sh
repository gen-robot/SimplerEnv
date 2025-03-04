# dpo
ckpt_path="ZijianZhang/OpenVLA-7B-SFT-Simpler"
for task in "PutSpoonOnTableClothInScene-v1" "PutCarrotOnPlateInScene-v1" "StackGreenCubeOnYellowCubeBakedTexInScene-v1" "PutEggplantInBasketScene-v1"; do
  CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/eval_ms3_collect_dpo.py \
    --model="openvla" --ckpt_path="${ckpt_path}" -e "${task}" \
    --openvla_unnorm_key="Simpler"
done