export CUDA_VISIBLE_DEVICES=1
python -m simpler_env.eval_ms3_visualize --model openvla --ckpt_path ZijianZhang/OpenVLA-7B-SFT-Simpler \
    --env_id "PandaPutCarrotOnPlateInScene-v1" --policy_setup panda \
    --max_episode_steps 200 --num_episodes 50 --num_envs 10 --save_video

# python -m simpler_env.eval_ms3_visualize --model openvla --ckpt_path openvla/openvla-7b \
#     --env_id "StackGreenCubeOnYellowCubeBakedTexInScene-v1" --policy_setup widowx_bridge