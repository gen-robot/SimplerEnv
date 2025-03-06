# export CUDA_VISIBLE_DEVICES=2
python -m simpler_env.eval_ms3_visualize --model openvla --ckpt_path openvla/openvla-7b \
    --env_id "PandaPutCarrotOnPlateInScene-v1" --policy_setup panda \
    --max_episode_steps 200 --num_episodes 100 --num_envs 10 --save_video

# python -m simpler_env.eval_ms3_visualize --model openvla --ckpt_path openvla/openvla-7b \
#     --env_id "StackGreenCubeOnYellowCubeBakedTexInScene-v1" --policy_setup widowx_bridge