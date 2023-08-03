import os
import time
import datetime
import random
import numpy as np
from collections import defaultdict
from pprint import pprint
from tabulate import tabulate
from typing import Any, Iterable

from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

from scml.oneshot import is_system_agent
from scml.oneshot.common import QUANTITY
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.rl.action import (
    ActionManager,
    DefaultActionManager
)
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.factory import (
    OneShotWorldFactory,
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
)
from scml.oneshot.rl.observation import (
    FixedPartnerNumbersObservationManager,
    LimitedPartnerNumbersObservationManager,
    ObservationManager
)
from scml.utils import anac2023_oneshot
from scml.oneshot.agents import GreedyOneShotAgent, SingleAgreementAspirationAgent, SyncRandomOneShotAgent

from scml_agents.scml2023 import QuantityOrientedAgent, CCAgent, KanbeAgent

from stable_baselines3 import A2C, PPO

from tqdm import tqdm

from util import format_time, get_dirname
from observation import BetterFixedPartnerNumbersObservationManager, DictBetterObservationManager

def test_tournament(
        agent : OneShotRLAgent,
        n_configs : int = 20,
        opponents : list[OneShotAgent] = [QuantityOrientedAgent, CCAgent, KanbeAgent],
        print_results : bool = True,
):
    """
    Basic script for testing a OneShotRLAgent in a tournament environment
    
    A OneShotRLAgent passed to this function should be able to handle arbitrary tournament worlds.
    """
    competitors = [agent] + opponents
    start = time.perf_counter()
    results = anac2023_oneshot(
        competitors=competitors,
        n_configs=n_configs,
    )
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[
        -1
    ]
    # display results
    if print_results:
        print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
        print(f"Finished in {format_time(time.perf_counter() - start)}")

    return results


def test_model(
        models : PPO | A2C | list[PPO | A2C],
        obs_manager_types : ObservationManager | list[ObservationManager],
        factory : OneShotWorldFactory,
        act_manager_types : ActionManager | list[ActionManager] = DefaultActionManager,
        n_trials : int = 20,
):
    """
    Basic script for testing trained models

    The input can either be a single model or a list of models. In the case of a list, each model can have its own observation and action manager,
    or a single one of either can be passed and an instance of the same manager will be used for each model.

    Args:
        models: A trained Stable Baselines 3 model, or a list of models
        obs_manager_types: The class name for the observation manager that the model will use, or a list of observation managers
        factory: An instance of a OneShotWorldFactory to be used to generate worlds for testing
        act_manager_types: The class name for the action manager that the model will use, or a list of action managers
        n_trials: The number of worlds to simulate
    """
    assert (not isinstance(obs_manager_types, list) or len(models) == len(obs_manager_types)) and (not isinstance(act_manager_types, list) or len(models) == len(act_manager_types))
    
    if not isinstance(models, list):
        models = [models]
    models = [model_wrapper(model) for model in models]

    if not isinstance(obs_manager_types, list):
        obs_manager_types = [obs_manager_types] * len(models)
    obs_managers = [obs_manager_type(factory=factory, extra_checks=False) for obs_manager_type in obs_manager_types]

    if not isinstance(act_manager_types, list):
        act_manager_types = [act_manager_types] * len(models)
    act_managers = [act_manager_type(factory=factory) for act_manager_type in act_manager_types]

    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()


    for _ in tqdm(range(n_trials)):
        world, agents = factory(
            types=(OneShotRLAgent,),
            params=(dict(models=models, observation_managers=obs_managers, action_managers=act_managers),),
        )
        world.run()

        all_scores = world.scores()
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            key = aid if n_trials == 1 else f"{aid}@{world.id[:4]}"
            agent_scores[key] = (
                 agent.type_name.split(':')[-1].split('.')[-1],
                 all_scores[aid],
                 '(bankrupt)' if world.is_bankrupt[aid] else ''
                )
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            type_ = agent.type_name.split(':')[-1].split('.')[-1]
            type_scores[type_] += all_scores[aid]
            counts[type_] += 1
    type_scores = {k: v/counts[k] if counts[k] else v for k, v in type_scores.items()}

    return world, agent_scores, type_scores


def analyze_contracts(world):
    """
    Analyzes the contracts signed in the given world
    """
    import pandas as pd
    data = pd.DataFrame.from_records(world.saved_contracts)
    return data.groupby(["seller_name", "buyer_name"])[["quantity", "unit_price"]].mean()


def print_agent_scores(agent_scores):
    """
    Prints scores of individiual agent instances
    """
    for aid, (type_, score, bankrupt) in agent_scores.items():
        print(f"Agent {aid} of type {type_} has a final score of {score} {bankrupt}")

def print_type_scores(type_scores):
    """Prints scores of agent types"""
    pprint(sorted(tuple(type_scores.items()), key=lambda x: -x[1]))



if __name__ == '__main__':
    models = [
        ('PPO_L0_4-partners_DICTOBS_20230731-110939_512000-steps', DictBetterObservationManager),
    ]
    for modelname, obsmanager in models:
        model = PPO.load(os.path.join(get_dirname(__file__), "models", modelname))

        factory = FixedPartnerNumbersOneShotFactory(n_consumers=4)
        
        world, ascores, tscores = test_model(
            models=model, 
            obs_manager_types=obsmanager, 
            factory=factory,
            n_trials=50,
        )
        print(f"Model {modelname} scores:")
        print_type_scores(tscores)
