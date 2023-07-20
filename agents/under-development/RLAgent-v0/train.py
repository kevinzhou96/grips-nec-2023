import os
import time

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse
from negmas.helpers import humanize_time

from scml.oneshot.common import QUANTITY
from scml.oneshot.rl.action import (
    ActionManager,
    FixedPartnerNumbersActionManager,
    LimitedPartnerNumbersActionManager,
    UnconstrainedActionManager,
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
)

from stable_baselines3 import A2C, PPO, DQN

from util import format_time, get_dirname

def make_training_env(
        level : int,
        n_partners : int,
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
        action_manager=FixedPartnerNumbersActionManager(factory=factory),
        observation_manager=FixedPartnerNumbersObservationManager(factory=factory, extra_checks=extra_checks),
        factory=factory,
        extra_checks=extra_checks,
    )

def train(
        level : int = 0,
        n_partners : int = 4,
        total_timesteps : int = 10_000,
        algorithm : str = "PPO",
        verbose : bool = True,
        logdir : str | None = None,
):
    env_train = make_training_env(level=level, n_partners=n_partners)

    alg = dict(
        PPO=PPO,
        A2C=A2C,
        DQN=DQN,
    )[algorithm]
    model = alg("MlpPolicy", env_train, verbose=verbose, tensorboard_log=logdir)

    if verbose:
        start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)

    model_name = f"{algorithm}_L{level}_{n_partners}-partners_{total_timesteps}-steps_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(os.path.join(get_dirname(__file__), "models", model_name))

    if verbose:
        print(f"Finished training in {format_time(time.perf_counter() - start)}")

if __name__ == '__main__':
    logdir = os.path.join(get_dirname(__file__), "logs", "")
    train(total_timesteps=100_000, logdir=logdir)