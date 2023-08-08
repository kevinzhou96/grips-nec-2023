"""install negmas and scml"""
pip install negmas==0.9.8
pip install scml==0.5.6


"""import"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import random
import csv
import shutil

import negmas
from negmas import Outcome

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
from scml.scml2020.utils import anac2023_collusion, anac2023_oneshot, anac2023_std
from tabulate import tabulate

from negmas import Contract, Outcome, ResponseType, SAOState


"""make environment"""
from collections import defaultdict
import random
from negmas import ResponseType
from scml.oneshot import *
from scml.scml2020 import is_system_agent
from pprint import pprint

def try_agent(agent_type, n_processes=2, draw=True):
    """Runs an agent in a world simulation against a randomly behaving agent"""
    return try_agents([RandomOneShotAgent, agent_type], n_processes, draw=draw)

def try_agents(agent_types, n_processes=2, n_trials=1, draw=True, agent_params=None):
    """
    Runs a simulation with the given agent_types, and n_processes n_trial times.
    Optionally also draws a graph showing what happened
    """
    type_scores = defaultdict(float)
    counts = defaultdict(int)
    agent_scores = dict()
    for _ in range(n_trials):
        p = n_processes if isinstance(n_processes, int) else random.randint(*n_processes)
        world = SCML2023OneShotWorld(
        **SCML2023OneShotWorld.generate(agent_types, agent_params=agent_params, n_steps=10,
                                        n_processes=p, random_agent_types=True),
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


"""define run function"""
def run(
    competition = "oneshot",
    reveal_names=True,
    n_steps=10,
    n_configs=5
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

    global competitors

    start = time.perf_counter()
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
