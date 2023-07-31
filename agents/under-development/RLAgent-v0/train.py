import os
import time
import sys
import argparse

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

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
from gymnasium import spaces

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from util import get_dirname
from observation import BetterFixedPartnerNumbersObservationManager, DictBetterObservationManager
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
        log_dir : str | None = None,
        checkpoint_freq : int = 0,
        pretrained : str | None = None,
        n_steps : int = 2048,
        n_envs : int = 1,
        env_kwargs : dict = dict(),
        model_kwargs : dict = dict(),
        learning_kwargs : dict = dict(),
        save_dir: str = "models",
        name_suffix: str | None = None,
        **kwargs,
):
    if n_envs  > 1:
        env_train = SubprocVecEnv([lambda: Monitor(make_training_env(level=level, n_partners=n_partners, obs_manager_type=obs_manager_type, reward_function=reward_function, **env_kwargs)) 
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
        if type(env_train.observation_space) == spaces.Dict:
            policy = "MultiInputPolicy"
        else:
            policy = "MlpPolicy"
        model = alg(policy, env_train, verbose=verbose, tensorboard_log=log_dir, n_steps=n_steps, **model_kwargs)
        model_name = f"{algorithm}_L{level}_{n_partners}-partners_{time.strftime('%Y%m%d-%H%M%S')}_{total_timesteps}-steps"
    
    if not name_suffix: model_name += name_suffix

    print(f"Training model {model_name}")

    if not os.path.isabs(save_dir):
        save_dir = os.path.join(get_dirname(__file__), save_dir)
    save_path = os.path.join(save_dir, model_name)    

    if checkpoint_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=(checkpoint_freq // n_envs),
            save_path=os.path.join(save_dir,"checkpoints", model_name),
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

    model.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", 
                        default=os.path.join("logs", ""),
                        help="specifies the directory to save TensorBoard log files. Default is 'logs/'", 
                        )
    parser.add_argument("--save_dir",
                        default=os.path.join("models", ""),
                        help="specifies the directory to save the model. Default is 'models/'")
    parser.add_argument("--steps-per-update", 
                        type=int, default=1024,
                        help="specifies the number of timesteps to run between policy updates. Default is 1024", 
                        )
    parser.add_argument("--n-updates", 
                        type=int, 
                        default=1000,
                        help="specifies the number of times the policy should be updated during training. Default is 1000", 
                        )
    parser.add_argument("--checkpoint-freq", 
                        help="specifies the frequency (in steps) for saving checkpoints. Default is total_timesteps / 10, with min frequency 20,480 and max frequency 102,400",
                        )
    parser.add_argument("--pretrained", 
                        help="specifies a file to load a pretrained model from for continued training",
                        )
    parser.add_argument("-v", "--verbose", 
                        action="store_true", 
                        help="toggles on verbose output for training",
                        )
    parser.add_argument("--no-progress-bar",
                        action="store_true",
                        help="toggles off the training progress bar",
                        )
    parser.add_argument("-a", "--algorithm",
                        choices=["A2C", "PPO"],
                        default="PPO",
                        help="specifies the learning algorithm. Currently allowed choices are A2C and PPO. Default is PPO",
                        )
    parser.add_argument("--n-envs",
                        type=int,
                        default=1,
                        help="specifies the number of parallel training environments. Default is 1 (no parallel training)",
                        )
    parser.add_argument("--learning-rate",
                        type=float,
                        default=1e-3,
                        help="sets the learning rate for training. Default is 1e-3")
    parser.add_argument("--name-suffix",
                        help="adds a suffix to the model name")
    args = parser.parse_args()


    if os.path.isabs(args.log_dir):
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(get_dirname(__file__), args.log_dir)

    total_timesteps = args.steps_per_update * args.n_updates

    if args.checkpoint_freq:
        checkpoint_freq = args.checkpoint_freq
    else:
        min_checkpoint_freq = 20_480
        max_checkpoint_freq = 102_400
        if total_timesteps < min_checkpoint_freq * 4:
            checkpoint_freq = 0
        else:
            checkpoint_freq = max(min_checkpoint_freq, min((n_updates // 10)* steps_per_update, max_checkpoint_freq))

    progress_bar = not args.no_progress_bar

    def exponential_decay(t, initial_value, final_value):
        return initial_value * (final_value / initial_value) ** (t)
    
    def truncated_linear_decay(t, initial_value, final_value, cutoff):
        slope = (final_value - initial_value) / cutoff
        return max(final_value, (t * slope) + 1)

    train(
        obs_manager_type=DictBetterObservationManager,
        # reward_function=ReducingNeedsReward(),
        algorithm=args.algorithm, 
        total_timesteps=total_timesteps, 
        log_dir=log_dir,
        progress_bar=progress_bar,
        verbose=args.verbose,
        checkpoint_freq=checkpoint_freq,
        pretrained=args.pretrained,
        n_steps=args.steps_per_update,
        n_envs=args.n_envs, 
        env_kwargs=dict(),
        model_kwargs={'learning_rate': args.learning_rate},
        learning_kwargs=dict(),
        save_dir=args.save_dir,
        name_suffix=args.name_suffix,
    )
