import os
import time

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

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

def make_oneshot_env(
        level : int = 0,
        n_consumers : int = 4,
        n_suppliers: int = 0,
        extra_checks : bool = False,
) -> OneShotEnv:
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
        n_consumers : int = 4,
        n_suppliers : int = 0,
        total_timesteps : int = 100_000,
        algorithm = "PPO",
):
    env_train = make_oneshot_env(level=level, n_consumers=n_consumers, n_suppliers=n_suppliers)

    alg = dict(
        PPO=PPO,
        A2C=A2C,
        DQN=DQN,
    )[algorithm]
    model = alg("MlpPolicy", env_train, verbose=1)
    model.learn(total_timesteps=total_timesteps)


    n_partners = n_consumers + n_suppliers
    model_name = f"{algorithm}_L{level}_{n_partners}-partners_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", model_name))

if __name__ == '__main__':
    train(total_timesteps=100)