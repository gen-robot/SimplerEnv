export CUDA_VISIBLE_DEVICES=4
python -m simpler_env.eval_ms3_task --model openvla --ckpt_path "openvla/openvla-7b" \
    --env_id "PushCube-v1" --policy_setup panda \
    --max_episode_steps 200 --num_episodes 50 --num_envs 10 --save_video --openvla_unnorm_key hah