import gymnasium as gym
import os
from .wrapper import (
    DoneOnSuccessWrapper,
    IOFromConfigWrapper,
    SeqActionFromConfigWrapper,
)
from homebot_sapien.vec_wrapper.subproc_vec_env import SubprocVecEnv
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


def make_env(
    env_id,
    done_when_success=False,
    io_config: dict = None,
    obs_normalize_params: dict = None,
    gamma: float = 1.0,
    return_middle_obs: bool = False,
    kwargs={},
):
    env = gym.make(env_id, **kwargs)
    # if flexible_time_limit:
    #     from utils.wrapper import FlexibleTimeLimitWrapper
    #     env = FlexibleTimeLimitWrapper(env)
    # if obs_keys is not None and isinstance(obs_keys, list):
    # env = FlattenObservation(env)
    if done_when_success:
        env = DoneOnSuccessWrapper(env)
    env = RecordEpisodeStatistics(env, deque_size=100)
    # if log_dir is not None:
    #     env = Monitor(env, os.path.join(log_dir, "%d.monitor.csv" % rank), info_keywords=info_keywords)
    if io_config is not None:
        # env = IOFromConfigWrapper(env, io_config)
        env = SeqActionFromConfigWrapper(
            env,
            io_config,
            obs_normalize_params,
            gamma=gamma,
            return_middle_obs=return_middle_obs,
        )
    return env


def make_vec_env(env_id, num_workers, reset_when_done=True, **kwargs):
    def make_env_thunk(i):
        return lambda: make_env(env_id, **kwargs)

    env = SubprocVecEnv(
        [make_env_thunk(i) for i in range(num_workers)], reset_when_done=reset_when_done
    )
    return env
