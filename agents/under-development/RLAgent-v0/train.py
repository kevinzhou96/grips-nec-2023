import os
import time

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse
from negmas.helpers import humanize_time

from scml.oneshot.common import QUANTITY
from scml.oneshot.rl.action import (
    ActionManager,
    DefaultActionManager,
)
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.factory import (
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
)
from scml.oneshot.rl.observation import (
    FixedPartnerNumbersObservationManager,
    LimitedPartnerNumbersObservationManager,
    ObservationManager
)
from scml.oneshot.rl.reward import DefaultRewardFunction, RewardFunction

import gymnasium as gym

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv

from util import format_time, get_dirname
from observation import BetterFixedPartnerNumbersObservationManager
from reward import (
    ReducingNeedsReward,
    QuantityBasedReward,
)

def make_training_env(
        level : int,
        n_partners : int,
        obs_manager_type : ObservationManager,
        reward_function : RewardFunction = DefaultRewardFunction(),
        extra_checks : bool = False,
) -> OneShotEnv:
    
    if level == 0:
        n_consumers = n_partners
        n_suppliers = 0
    else:
        n_consumers = 0
        n_suppliers = n_partners

    factory = FixedPartnerNumbersOneShotFactory(
        n_suppliers=n_suppliers,
        n_consumers=n_consumers,
        level=level,
    )

    return OneShotEnv(
        action_manager=DefaultActionManager(factory=factory),
        observation_manager=obs_manager_type(factory=factory, extra_checks=extra_checks),
        factory=factory,
        extra_checks=extra_checks,
        reward_function=reward_function,
    )

def train(
        level : int = 0,
        n_partners : int = 4,
        obs_manager_type : ObservationManager = FixedPartnerNumbersObservationManager,
        reward_function : RewardFunction = DefaultRewardFunction(),
        total_timesteps : int = 10_000,
        algorithm : str = "PPO",
        progress_bar : bool = True,
        verbose : bool = False,
        logdir : str | None = None,
        checkpoint_freq : int = 0,
        pretrained : str | None = None,
        n_steps : int = 2048,
        n_envs : int = 1,
        env_kwargs : dict = dict(),
        learning_kwargs : dict = dict(),
        **kwargs,
):
    if n_envs  > 1:
        env_train = SubprocVecEnv([lambda: make_training_env(level=level, n_partners=n_partners, obs_manager_type=obs_manager_type, reward_function=reward_function, **env_kwargs) 
                                   for _ in range(n_envs)])
    else:
        env_train = make_training_env(level=level, n_partners=n_partners, obs_manager_type=obs_manager_type, reward_function=reward_function, **env_kwargs)

    alg = dict(
        PPO=PPO,
        A2C=A2C,
        DQN=DQN,
    )[algorithm]
    
    if pretrained:
        model = alg.load(pretrained, env=env_train)
        model_name = f"{os.path.basename(pretrained)}_{total_timesteps}-additional-steps"
    else:
        model = alg("MlpPolicy", env_train, verbose=verbose, tensorboard_log=logdir, n_steps=n_steps)
        model_name = f"{algorithm}_L{level}_{n_partners}-partners_{time.strftime('%Y%m%d-%H%M%S')}_{total_timesteps}-steps"

    if checkpoint_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(get_dirname(__file__),"models","checkpoints", model_name),
            name_prefix=model_name,
        )
        print(f"Checkpointing every {checkpoint_freq} iters")
    else:
        checkpoint_callback = None

    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name=model_name, 
        progress_bar=progress_bar,
        callback=checkpoint_callback,
        **learning_kwargs,
    )

    model.save(os.path.join(get_dirname(__file__), "models", model_name))

if __name__ == '__main__':
    logdir = os.path.join(get_dirname(__file__), "logs", "")

    steps_per_update = 1024
    n_updates = 1000
    total_timesteps = steps_per_update * n_updates

    min_checkpoint_freq = 20_480
    max_checkpoint_freq = 102_400

    if total_timesteps < min_checkpoint_freq * 4:
        checkpoint_freq = 0
    else:
        checkpoint_freq = max(min_checkpoint_freq, min((n_updates // 10)* steps_per_update, max_checkpoint_freq))

    # pretrained = os.path.join(get_dirname(__file__), "models", "checkpoints", "PPO_L0_4-partners_102400-steps_20230724-122807_61440_steps")
    pretrained = None

    train(
        obs_manager_type=BetterFixedPartnerNumbersObservationManager,
        reward_function=QuantityBasedReward(),
        algorithm="PPO", 
        total_timesteps=total_timesteps, 
        logdir=logdir,
        progress_bar=True,
        verbose=False,
        checkpoint_freq=checkpoint_freq,
        pretrained=pretrained,
        n_steps=steps_per_update,
        # n_envs=4, # CAUTION: USING MULTIPLE ENVS REQUIRES A CHANGE TO THE ENVIRONMENT REGISTRATION
        env_kwargs=dict(),
        learning_kwargs={'learning_rate' : 1e-3},
    )
