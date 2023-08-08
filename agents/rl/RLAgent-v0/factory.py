import random
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from attr import define, field
from pprint import pprint

from scml.common import intin, make_array
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import OneShotDummyAgent
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.world import (
    SCML2020OneShotWorld,
    SCML2021OneShotWorld,
    SCML2022OneShotWorld,
    SCML2023OneShotWorld,
)

from scml.oneshot.agents import (
    GreedyOneShotAgent,
    SingleAgreementAspirationAgent,
    SyncRandomOneShotAgent,
)
from scml.oneshot.rl.common import WorldFactory, isin
from scml.oneshot.rl.factory import OneShotWorldFactory


DefaultAgentsOneShot2023 = [
    GreedyOneShotAgent,
    SingleAgreementAspirationAgent,
    SyncRandomOneShotAgent,
]


@define(frozen=True)
class FixedPartnerNumberWithOpponentsOneShotFactory(OneShotWorldFactory):
    """Generates a 2023 OneShot world fixing the agent level, production capacity, and the number of partners,
    and optionally specifying some opponent tournament agents."""

    year: int = 2023
    level: int = 0
    n_processes: int = 2
    n_partners: int = 4
    # n_consumers: int = 4
    # n_suppliers: int = 0
    n_competitors: tuple[int, int] | int = (3, 7)
    # n_agents_per_level: tuple[int, int] | int = (4, 8)
    n_lines: tuple[int, int] | int = 10
    opponent_agents : list[str | type[OneShotAgent]] | None = None
    opponent_params : list[dict[str, Any]] | None = None
    non_competitors: list[str | type[OneShotAgent]] = DefaultAgentsOneShot2023

    def __attrs_post_init__(self):
        # assert self.level != 0 or self.n_suppliers == 0
        # assert (
        #     not isinstance(self.n_processes, int)
        #     or (self.level != self.n_processes - 1 and self.level != -1)
        #     or self.n_consumers == 0
        # )
        # if isinstance(self.n_processes, tuple):
        #     assert not (self.level > 0 and self.level < self.n_processes[-1] - 1) or (
        #         self.n_suppliers > 0 and self.n_consumers > 0
        #     )
        #     assert self.level == -1 or self.level < self.n_processes[-1]
        # else:
        #     assert not (self.level > 0 and self.level < self.n_processes - 1) or (
        #         self.n_suppliers > 0 and self.n_consumers > 0
        #     )
        #     assert self.level == -1 or self.level < self.n_processes
        assert self.level == 0 or self.level == 1
        if self.opponent_agents and not self.opponent_params:
            object.__setattr__(self, "opponent_params", [dict() for _ in self.opponent_agents])
        if self.opponent_agents:
            assert len(self.opponent_agents) == len(self.opponent_params)

    def make(
        self,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
        params: tuple[dict[str, Any], ...] | None = None,
    ) -> SCML2020OneShotWorld:
        """Generates a world"""
        # initialize parameters
        if types and params is None:
            params = tuple(dict() for _ in types)
        n_processes = intin(self.n_processes)
        n_lines = intin(self.n_lines)
        n_competitors = intin(self.n_competitors)

        # initialize list of agents
        n_agents = self.n_partners + n_competitors + len(types)
        agent_types = list(random.choices(self.non_competitors, k=n_agents))
        agent_params = None
        if params or self.opponent_params:
            agent_params: list[dict[str, Any]] | None = [dict() for _ in agent_types]
        
        # assign agents to levels
        agent_processes = np.ones(n_agents, dtype=int)
        if self.level == 0:
            agent_processes[:n_competitors+1] = 0
        else:
            agent_processes[:self.n_partners] = 0

        # select a random subset of opponents
        opponents = []
        n_opponents = 0
        if self.opponent_agents:
            n_opponents = random.randint(1, min(len(self.opponent_agents), n_agents-1))
            opponent_idxs = random.sample(range(len(self.opponent_agents)), n_opponents)
            opponents = np.asarray(self.opponent_agents)[opponent_idxs].tolist()
            opponent_params = np.asarray(self.opponent_params)[opponent_idxs].tolist()

        # assign given agents to factories
        if self.level == 0:
            my_idxs= random.sample(range(0, n_competitors+1), len(types))
        else:
            my_idxs = random.sample(range(n_competitors+2, n_agents), len(types))

        for idx, agent_type, p in zip(my_idxs, types, params):
            agent_types[idx] = agent_type
            if params or self.opponent_params:
                agent_params[idx]["controller_params"] = p

        # assign opponent agents to factories
        available_idxs = list(set(range(0,n_agents)) - set(my_idxs))
        opponent_idxs = random.sample(available_idxs, n_opponents)
        for idx, agent_type, p in zip(opponent_idxs, opponents, opponent_params):
            agent_types[idx] = agent_type
            if params or self.opponent_params:
                agent_params[idx]["controller_params"] = p


        return SCML2023OneShotWorld(
            **SCML2023OneShotWorld.generate(
                n_lines=n_lines,
                agent_types=agent_types,
                agent_params=agent_params,
                agent_processes=agent_processes,
                n_processes=n_processes,
                random_agent_types=False,
            ),
            one_offer_per_step=True,
            **self.world_params,
        )

    # TODO: implement is_valid_world()
    def is_valid_world(
        self,
        world: SCML2020OneShotWorld,
        types: tuple[type[OneShotAgent], ...] = (OneShotDummyAgent,),
    ) -> bool:
        """Checks that the given world could have been generated from this factory"""
        # if isinstance(types, OneShotAgent):
        #     types = (types,)  # type: ignore
        # for agent_type in types:
        #     expected_type = agent_type._type_name()
        #     agents = [
        #         i
        #         for i, type_ in enumerate(world.agent_types)
        #         if type_.split(":")[-1] == expected_type
        #     ]
        #     assert (
        #         len(agents) == 1
        #     ), f"Found the following agent of type {agent_type}: {agents}"
        #     agent: OneShotAgent = None  # type: ignore
        #     for a in world.agents.values():
        #         if a.type_name.split(":")[-1] == expected_type:
        #             agent = a  # type: ignore
        #             break
        #     else:
        #         warnings.warn(f"cannot find any agent of type {expected_type}")
        #         return False
        #     if not isin(world.n_processes, self.n_processes):
        #         warnings.warn(
        #             f"Invalid n_processes: {world.n_processes=} != {self.n_processes=}"
        #         )
        #         return False
        #     if not isin(agent.awi.n_lines, self.n_lines):
        #         warnings.warn(
        #             f"Invalid n_lines: {agent.awi.n_lines=} != {self.n_lines=}"
        #         )
        #         return False
        # # TODO: check non-competitor types
        # return self.is_valid_awi(agent.awi)  # type: ignore
        return True

    # TODO: implement is_valid_awi
    def is_valid_awi(self, awi: OneShotAWI) -> bool:
        # find my level
        # my_level = awi.n_processes - 1 if self.level < 0 else self.level
        # n_partners = self.n_consumers + self.n_suppliers
        # if not isin(len(awi.my_partners), n_partners):
        #     warnings.warn(
        #         f"Invalid n_partners: {len(awi.my_partners)=} != {n_partners=}"
        #     )
        #     return False
        # if my_level == 0:
        #     if not isin(len(awi.my_consumers), self.n_consumers):
        #         warnings.warn(
        #             f"Invalid n_consumers: {len(awi.my_consumers)=} != {self.n_consumers=}"
        #         )
        #         return False
        #     if len(awi.my_suppliers) != 1:
        #         warnings.warn(f"Invalid n_suppliers: {len(awi.my_suppliers)=} != 1")
        #         return False
        # elif my_level == awi.n_processes - 1:
        #     if not isin(len(awi.my_suppliers), self.n_suppliers):
        #         warnings.warn(
        #             f"Invalid n_suppliers: {len(awi.my_suppliers)=} != {self.n_suppliers=}"
        #         )
        #         return False
        #     if len(awi.my_consumers) != 1:
        #         warnings.warn(f"Invalid n_conumsers: {len(awi.my_consumers)=} != 1")
        #         return False
        # else:
        #     if not isin(len(awi.my_suppliers), self.n_suppliers):
        #         warnings.warn(
        #             f"Invalid n_suppliers: {len(awi.my_suppliers)=} != {self.n_suppliers=}"
        #         )
        #         return False
        #     if not isin(len(awi.my_consumers), self.n_consumers):
        #         warnings.warn(
        #             f"Invalid n_consumers: {len(awi.my_consumers)=} != {self.n_consumers=}"
        #         )
        #         return False
        # n_competitors = awi.n_competitors
        # if not isin(n_competitors, self.n_competitors):
        #     warnings.warn(
        #         f"Invalid n_competitors: {awi.n_competitors=} != {n_competitors=}"
        #     )
        #     return False
        return True

    # TODO: implement contains_factory()
    def contains_factory(self, factory: WorldFactory) -> bool:
        """Checks that the any world generated from the given `factory` could have been generated from this factory"""
        # if not isinstance(factory, self.__class__):
        #     return False
        # if not isin(factory.year, self.year):
        #     return False
        # if not isin(factory.n_processes, self.n_processes):
        #     return False
        # if not isin(factory.level, self.level):
        #     return False
        # if not isin(factory.n_consumers, self.n_consumers):
        #     return False
        # if not isin(factory.n_suppliers, self.n_suppliers):
        #     return False
        # if not isin(factory.n_competitors, self.n_competitors):
        #     return False
        # if not isin(factory.n_agents_per_level, self.n_agents_per_level):
        #     return False
        # if not isin(factory.n_lines, self.n_lines):
        #     return False
        # if set(factory.non_competitors).difference(list(self.non_competitors)):
        #     return False
        return True