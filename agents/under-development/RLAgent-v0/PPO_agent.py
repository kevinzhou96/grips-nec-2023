#!/usr/bin/env python
"""
**Submitted to ANAC 2023 SCML (OneShot track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2023 SCML.
"""
from __future__ import annotations

# used to repeat the response to every negotiator.
import itertools

# required for running tournaments and printing
import time

# required for typing
from typing import Any

from negmas.helpers import humanize_time
from negmas.sao import SAOResponse, SAOState

# required for development
from scml.oneshot import OneShotAWI, OneShotSyncAgent
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.utils import anac2023_collusion, anac2023_oneshot, anac2023_std
from scml.oneshot.agents import RandomOneShotAgent, SyncRandomOneShotAgent, GreedyOneShotAgent, SingleAgreementAspirationAgent, GreedySingleAgreementAgent, GreedySyncAgent
from scml_agents.scml2023 import QuantityOrientedAgent
from scml.oneshot import is_system_agent, SCML2023OneShotWorld
from scml.oneshot.rl.action import FixedPartnerNumbersActionManager
from scml.oneshot.rl.observation import FixedPartnerNumbersObservationManager
from scml.oneshot.rl.factory import FixedPartnerNumbersOneShotFactory

from tabulate import tabulate

from negmas import Contract, Outcome, ResponseType, SAOState
from negmas.helpers import humanize_time
from negmas.sao import SAOResponse, SAOState

from collections import defaultdict
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pprint import pprint
from tqdm import tqdm
import random
import os

from stable_baselines3 import PPO


class PPOAgent(OneShotSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `OneShotUFun` in the docs for more details).
    """

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the
              same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        obs = self.obs_manager.encode(state=self.awi.state)
        action, _ = self.policy.predict(obs)
        offers = self.act_manager.decode(awi=self.awi, action=action)
        return offers


    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        return dict()

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""
        model_path = os.path.join(os.getcwd(), "4partners-l0-agent")
        self.policy = PPO.load(model_path)

        self.factory = FixedPartnerNumbersOneShotFactory()
        self.obs_manager = FixedPartnerNumbersObservationManager(factory=self.factory, extra_checks=False)
        self.act_manager = FixedPartnerNumbersActionManager(factory=self.factory)

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""

    def step(self):
        """Called at at the END of every production step (day)"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""


def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=50,
    n_configs=20,
    competitors=None,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are oneshot, std,
                     collusion).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """
    if not competitors:
        if competition == "oneshot":
            competitors = [PPOAgent, RandomOneShotAgent, SyncRandomOneShotAgent]
            # competitors = [UfunQuantityOrientedAgent, QuantityOrientedAgent]
        else:
            from scml.scml2020.agents import BuyCheapSellExpensiveAgent, DecentralizingAgent

            competitors = [
                PPOAgent,
                DecentralizingAgent,
                BuyCheapSellExpensiveAgent,
            ]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2023_std
    elif competition == "collusion":
        runner = anac2023_collusion
    else:
        runner = anac2023_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[
        -1
    ]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


def try_agent(agent_type, n_processes=2, n_trials=1, n_steps=None, draw=True):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    return try_agents([GreedySyncAgent, SingleAgreementAspirationAgent, agent_type, RandomOneShotAgent, GreedyOneShotAgent], n_processes, n_trials, n_steps=n_steps, draw=draw)

def try_agents(agent_types, n_processes=2, n_trials=1, n_steps=20, draw=True, agent_params=None):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    for _ in tqdm(range(n_trials)):
        p = n_processes if isinstance(n_processes, int) else random.randint(*n_processes)
        world = SCML2023OneShotWorld(
        **SCML2023OneShotWorld.generate(agent_types, agent_params=agent_params, n_steps=n_steps,
                                        n_processes=p, random_agent_types=True, n_agents_per_process=4),
        construct_graphs=True,
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
    if draw:
        world.draw(
            what=["contracts-concluded"],
            steps=(0, world.n_steps - 1),
            together=True, ncols=1, figsize=(20, 20)
        )
        plt.show()

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

def run_single_type_world(agent_type, n_trials=10, n_steps=50):
    world, ascores, tscores = try_agents([agent_type], draw=False, n_processes=2, n_trials=n_trials, n_steps=n_steps)
    # print_agent_scores(ascores)
    print_type_scores(tscores)


if __name__ == "__main__":
    import sys

    # run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
    # run("oneshot", n_steps=20, n_configs=5, competitors=[PPOAgent, GreedyOneShotAgent, SingleAgreementAspirationAgent, QuantityOrientedAgent])

    world, ascores, tscores = try_agent(PPOAgent, n_processes=2, n_trials=5, n_steps=50, draw=False)
    print_type_scores(tscores)
    # print_agent_scores(ascores)

    # run_single_type_world(PPOAgent, n_trials=20, n_steps=100)
    # run_single_type_world(QuantityOrientedAgent)
