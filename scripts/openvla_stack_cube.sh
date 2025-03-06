# export CUDA_VISIBLE_DEVICES=2
python -m simpler_env.eval_ms3_visualize --model openvla --ckpt_path ZijianZhang/OpenVLA-7B-SFT-Simpler \
    --env_id "PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1" --policy_setup panda \
    --max_episode_steps 300 --num_episodes 100 --num_envs 10 --save_video

# python -m simpler_env.eval_ms3_visualize --model openvla --ckpt_path ZijianZhang/OpenVLA-7B-SFT-Simpler \
#     --env_id "StackGreenCubeOnYellowCubeBakedTexInScene-v1" --policy_setup widowx_bridge