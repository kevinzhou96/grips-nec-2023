import os
import time
import datetime
import random
import numpy as np
from collections import defaultdict
from pprint import pprint

from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse
from negmas.helpers import humanize_time

from scml.oneshot import is_system_agent
from scml.oneshot.common import QUANTITY
from scml.oneshot.rl.action import (
    ActionManager,
    DefaultActionManager
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
from scml.utils import anac2023_oneshot
from scml.oneshot.agents import GreedyOneShotAgent, SingleAgreementAspirationAgent, SyncRandomOneShotAgent

from scml_agents.scml2023 import QuantityOrientedAgent, CCAgent, KanbeAgent

from stable_baselines3 import A2C, PPO, DQN

from tqdm import tqdm

from util import format_time, get_dirname
from observation import BetterFixedPartnerNumbersObservationManager

# def test_tournament(
#         agent : OneShotRLAgent,
#         n_steps : tuple[int, int] | int = (50,200),
#         n_configs : int = 20
# ):
#     competitors = [agent, QuantityOrientedAgent, CCAgent]
#     start = time.perf_counter()
#     if isinstance(n_steps, tuple):
#         n_steps = random.randint(n_steps[0], n_steps[1])
#     results = anac2023_oneshot(
#         competitors=competitors,
#         n_steps=n_steps,
#         n_configs=n_configs,
#     )
#     results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
#         "."
#     ).str[
#         -1
#     ]
#     # display results
#     print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
#     print(f"Finished in {humanize_time(time.perf_counter() - start)}")


def test(
        model,
        obs_manager_type,
        level : int,
        n_partners : int,
        n_trials : int = 20,
):
    if level == 0:
        n_consumers = n_partners
        n_suppliers = 0
    else:
        n_consumers = 0
        n_suppliers = n_partners
    
    factory = FixedPartnerNumbersOneShotFactory(level=level, n_consumers=n_consumers, n_suppliers=n_suppliers)
    obs_manager = obs_manager_type(factory=factory, extra_checks=False)
    act_manager = DefaultActionManager(factory=factory)

    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()

    for _ in tqdm(range(n_trials)):
        world, agents = factory(
            types=(OneShotRLAgent,),
            params=(dict(models=[model_wrapper(model)], observation_managers=[obs_manager], action_managers=[act_manager]),),
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
        # 'PPO_L0_4-partners_10000-steps_20230721-122249',
        # ('PPO_L0_4-partners_20230726-141236_102400-steps', BetterFixedPartnerNumbersObservationManager),
        ('PPO_L0_4-partners_20230726-144321_102400-steps', BetterFixedPartnerNumbersObservationManager),
    ]
    for modelname, obsmanager in models:
        model = PPO.load(os.path.join(get_dirname(__file__), "models", modelname))
        
        world, ascores, tscores = test(
            model=model, 
            obs_manager_type=obsmanager,
            level=0, 
            n_partners=4, 
            n_trials=40,
        )
        print(f"Model {modelname} scores:")
        print_type_scores(tscores)
        # print(analyze_contracts(world))