import os
import time
import sys
import argparse
from typing import Any

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
    """
    Creates a OneShotEnv for use in training.
    """
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
        verbose : int = 1,
        log_dir : str | None = None,
        checkpoint_freq : int = 0,
        pretrained : str | None = None,
        n_steps : int = 2048,
        n_envs : int = 4,
        env_kwargs : dict = dict(),
        model_kwargs : dict = dict(),
        learning_kwargs : dict = dict(),
        save_dir: str = "models",
        name_suffix: str | None = None,
        **kwargs,
) -> PPO | A2C:
    """
    Trains and saves an RL model for an SCML OneShot negotiating agent.
    
    Args:
        level: The level (process) in which the agent will be placed
        n_partners: The number of negotiating partners the agent will have in the other level
        obs_manager_type: The observation manager for the training environment. Note that this argument should be passed as a subclass of ObservationManager
        reward_function: The reward function for the training environment. Note that this argument should be passed as an instance of a RewardFunction
        total_timesteps: The total number of timesteps to run the training algorithm
        algorithm: The RL algorithm to be used for training, given as a string. Currently, PPO and A2C are supported
        progress_bar: Whether or not to display a progress bar during training.
        verbose: Controls the verbosity of output during training. 0 prints nothing, 1 prints basic information about the model, 2 turns on 
                 verbosity for Stable Baselines 3 functions. Default is 1
        log_dir: Specifies the directory (either absolute, or relative to the location of train.py) for saving TensorBoard logs
        checkpoint_freq: The frequency (in timesteps) at which checkpoints should be saved. The default value of 0 means no checkpoints will be saved
        pretrained: Specifies the name of a zip folder containing a previously trained model, to be used as a starting point for further training
        n_steps: The number of timesteps to run for each environment per update. Note that the default value of 2048 is the Stable Baselines 3 default value
                 for PPO. The default value for A2C is much lower, however, it is likely worth considering the number of steps that are likely to occur in a 
                 single run of an SCML world in order to best decide the value of n_steps
        n_envs: The number of environments to run in parallel during training
        env_kwargs: Any keyword arguments that should be passed to the environment creation function
        model_kwargs: Any keyword arguments that should be passed to the model creation function 
        learning_kwargs: Any keyword arguments that should be passed to the learning function
        save_dir: Specifies the directory (either absolute, or relative to the location of train.py) for saving the final trained model
        name_suffx: A suffix to append onto the model name (usually used to attach additional notes about the model)

    Returns:
        The trained agent
    """

    sb3verbose = True if verbose == 2 else False

    # create the training environment
    if n_envs  > 1:
        env_train = SubprocVecEnv([lambda: Monitor(make_training_env(level=level, 
                                                                     n_partners=n_partners, 
                                                                     obs_manager_type=obs_manager_type, 
                                                                     reward_function=reward_function, 
                                                                     **env_kwargs,
                                                                     )) 
                                   for _ in range(n_envs)])
    else:
        env_train = make_training_env(level=level, 
                                      n_partners=n_partners, 
                                      obs_manager_type=obs_manager_type, 
                                      reward_function=reward_function, 
                                      **env_kwargs,
                                      )

    alg = dict(
        PPO=PPO,
        A2C=A2C,
    )[algorithm]
    
    # load a pretrained model, or create a new one 
    if pretrained:
        model = alg.load(pretrained, env=env_train)
        model_name = f"{os.path.basename(pretrained)}_{total_timesteps}-additional-steps"
    else:
        # Dict spaces require a MultiInputPolicy
        if type(env_train.observation_space) == spaces.Dict:
            policy = "MultiInputPolicy"
        else:
            policy = "MlpPolicy"
        model = alg(policy, env_train, verbose=sb3verbose, tensorboard_log=log_dir, n_steps=n_steps, **model_kwargs)
        model_name = f"{algorithm}_L{level}_{n_partners}-partners_{time.strftime('%Y%m%d-%H%M%S')}_{total_timesteps}-steps"
    if not name_suffix: model_name += name_suffix

    if verbose:
        print(f"Training model {model_name}")

    if not os.path.isabs(save_dir):
        save_dir = os.path.join(get_dirname(__file__), save_dir)
    save_path = os.path.join(save_dir, model_name)    

    # activate checkpointing. If parallel training is used, save_freq must be scaled to account for one step parallel step being equivalent to n_envs individual steps
    if checkpoint_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=(checkpoint_freq // n_envs),
            save_path=os.path.join(save_dir,"checkpoints", model_name),
            name_prefix=model_name,
        )
        if verbose:
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
    return model

def parse_args(default_values : dict[str, Any]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", 
                        default=default_values["log_dir"],
                        help="specifies the directory to save TensorBoard log files. Default is 'logs/'", 
                        )
    parser.add_argument("--save-dir",
                        default=default_values["save_dir"],
                        help="specifies the directory to save the model. Default is 'models/'")
    parser.add_argument("--steps-per-update", 
                        type=int, 
                        default=default_values["steps_per_update"],
                        help="specifies the number of timesteps to run between policy updates. Default is 1024", 
                        )
    parser.add_argument("--n-updates", 
                        type=int, 
                        default=default_values["n_updates"],
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
                        default=default_values["algorithm"],
                        help="specifies the learning algorithm. Currently allowed choices are A2C and PPO. Default is PPO",
                        )
    parser.add_argument("--n-envs",
                        type=int,
                        default=default_values["n_envs"],
                        help="specifies the number of parallel training environments. Default is 4",
                        )
    parser.add_argument("--learning-rate",
                        type=float,
                        default=default_values["learning_rate"],
                        help="sets the learning rate for training. Default is 1e-3")
    parser.add_argument("--name-suffix",
                        help="adds a suffix to the model name")
    
    return parser.parse_args()


if __name__ == '__main__':
    default_values = {
        "log_dir": os.path.join("logs", ""),
        "save_dir": os.path.join("models", ""),
        "steps_per_update": 1024,
        "n_updates": 1000,
        "algorithm": "PPO",
        "n_envs": 4,
        "learning_rate": 1e-3,
    }
    args = parse_args(default_values=default_values)
    
    if os.path.isabs(args.log_dir):
        log_dir = args.log_dir
    # make log_dir relative to location of train.py
    else:
        log_dir = os.path.join(get_dirname(__file__), args.log_dir)

    total_timesteps = args.steps_per_update * args.n_updates

    if args.checkpoint_freq:
        checkpoint_freq = args.checkpoint_freq
    # default checkpointing behaviour is to scale checkpoint frequency with total number of timesteps, with min frequency 20,480 and max frequency 102,400
    else:
        min_checkpoint_freq = 20_480
        max_checkpoint_freq = 102_400
        if total_timesteps < min_checkpoint_freq * 4:
            checkpoint_freq = 0
        else:
            checkpoint_freq = max(min_checkpoint_freq, min((args.n_updates // 10)* args.steps_per_update, max_checkpoint_freq))

    progress_bar = not args.no_progress_bar

    # decay functions that can be used to vary the learning rate. Note that the Stable Baselines learning_rate input parameter represents
    # the amount of time remaining, so decay functions should be passed to train() as lambda t: decay_fn(1-t, ...)
    def exponential_decay(t, initial_value, final_value):
        return initial_value * (final_value / initial_value) ** (t)
    
    def truncated_linear_decay(t, initial_value, final_value, cutoff):
        slope = (final_value - initial_value) / cutoff
        return max(final_value, (t * slope) + 1)

    # Since the obs_manager_type and reward_function arguments are a class and an instance of a class, these parameters are not handled through 
    # command line arguments, and so should be specified directly in the call to train()
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
