"""
A first attempt at a new observation manager for RLAgent-v0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from negmas import SAOResponse, ResponseType

import numpy as np
from attr import define, field
from gymnasium import spaces
from negmas.helpers.strings import itertools

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import NegotiationDetails, OneShotState
from scml.oneshot.rl.common import isin
from scml.oneshot.rl.factory import (
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
    OneShotWorldFactory,
)
from scml.oneshot.rl.observation import ObservationManager, BaseObservationManager
from scml.scml2019.common import QUANTITY, UNIT_PRICE

@define(frozen=True)
class BetterFixedPartnerNumbersObservationManager(BaseObservationManager):
    n_bins: int = 10
    n_sigmas: int = 2
    extra_checks: bool = True
    n_prices: int = 2
    negotiation_limit : int = 40
    n_partners: int = field(init=False)
    n_suppliers: int = field(init=False)
    n_consumers: int = field(init=False)
    max_quantity: int = field(init=False)
    n_lines: int = field(init=False)

    def __attrs_post_init__(self):
        assert isinstance(self.factory, FixedPartnerNumbersOneShotFactory)
        object.__setattr__(self, "n_suppliers", self.factory.n_suppliers)
        object.__setattr__(self, "n_consumers", self.factory.n_consumers)
        object.__setattr__(self, "max_quantity", self.factory.n_lines)
        object.__setattr__(self, "n_lines", self.factory.n_lines)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> spaces.Space:
        """
        Creates the observation space
        
        The observation space consists of:
            - The quantity and price for each incoming offer
            - The current needed quantity
            - The current round of negotations
            - The number of competitors
            - The current relative simulation time
            - The curent disposal cost
            - The current shortfall penalty
            - The relative price increase from input to output product
        """
        if isinstance(self.factory.n_agents_per_level, tuple):
            max_agents_per_level = self.factory.n_agents_per_level[-1]
        else:
            max_agents_per_level = self.factory.n_agents_per_level
        if isinstance(self.factory.n_competitors, tuple):
            max_n_competitors = self.factory.n_competitors[-1]
        else:
            max_n_competitors = self.factory.n_competitors
        return spaces.MultiDiscrete(
            np.asarray(
                list(
                    itertools.chain(
                        [self.max_quantity + 1, self.n_prices] * self.n_partners
                    )
                )
                + [self.max_quantity + 1]
                + [self.negotiation_limit + 1]
                + [max(max_agents_per_level, max_n_competitors) + 1]
                + [self.n_bins + 1] * 4
            ).flatten()
        )
    
    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes the state as an array"""
        partners = state.my_partners
        partner_index = dict(zip(partners, range(len(partners))))
        infos: list[NegotiationDetails] = [None] * len(partners)  # type: ignore
        negotiation_round = 0
        neg_relative_time = 0.0
        partners = list(
            itertools.chain(
                state.current_negotiation_details["buy"].items(),
                state.current_negotiation_details["sell"].items(),
            )
        )
        for partner, info in partners:
            infos[partner_index[partner]] = info
            negotiation_round = max(negotiation_round, info.nmi.state.step)
            neg_relative_time = max(neg_relative_time, info.nmi.state.relative_time)
        offers = [
            (0, 0)
            if info is None or info.nmi.state.current_offer is None  # type: ignore
            else (
                int(info.nmi.state.current_offer[QUANTITY]),  # type: ignore
                int(info.nmi.state.current_offer[UNIT_PRICE] - info.nmi.outcome_space.issues[UNIT_PRICE].min_value),  # type: ignore
            )
            for info in infos
        ]
        if self.extra_checks:
            assert (
                len(partners) == self.n_partners
            ), f"{len(partners)=} while {self.n_partners=}: {partners=}"
            assert (
                len(infos) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {infos=}"
            assert (
                len(offers) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {offers=}"

        # TODO add more state values here and remember to add corresponding limits in the make_space function
        def _normalize(x, mu, sigma, n_sigmas=self.n_sigmas):
            """
            Normalizes x between 0 and 1 given that it is sampled from a normal (mu, sigma).
            This is actually a very stupid way to do it.
            """
            mn = mu - n_sigmas * sigma
            mx = mu + n_sigmas * sigma
            if abs(mn - mx) < 1e-6:
                return 1
            return max(0, min(1, (x - mn) / (mx - mn)))

        extra = [
            max(0, state.needed_sales if state.level == 0 else state.needed_supplies),
            negotiation_round,
            state.n_competitors,
            int(state.relative_simulation_time * self.n_bins + 0.5),
            int(
                _normalize(
                    state.disposal_cost,
                    state.profile.disposal_cost_mean,
                    state.profile.disposal_cost_dev,
                )
                * self.n_bins
                + 0.5
            ),
            int(
                _normalize(
                    state.shortfall_penalty,
                    state.profile.shortfall_penalty_mean,
                    state.profile.shortfall_penalty_dev,
                )
                * self.n_bins
                + 0.5
            ),
            int(
                self.n_bins
                * max(
                    1,
                    (
                        state.trading_prices[state.my_output_product]
                        - state.trading_prices[state.my_input_product]
                    )
                    / state.trading_prices[state.my_output_product],
                )
                + 0.5
            ),
        ]

        v = np.asarray(np.asarray(offers).flatten().tolist() + extra)
        if self.extra_checks:
            space = self.make_space()
            assert space is not None and space.shape is not None
            exp = space.shape[0]
            assert (
                len(v) == exp
            ), f"{len(v)=}, {len(extra)=}, {len(offers)=}, {exp=}, {self.n_partners=}\n{state.current_negotiation_details=}"
            assert all(
                -1 < a < b for a, b in zip(v, space.nvec)  # type: ignore
            ), f"{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (state.exogenous_input_quantity , state.total_supplies , state.total_sales , state.exogenous_output_quantity) }"  # type: ignore

        return v
    
    def get_offers(self, awi: OneShotAWI, encoded : dict[str, np.ndarray | int | float]) -> dict[str, SAOResponse]:
        offers = encoded[: 2 * self.n_partners].reshape((self.n_partners, 2))
        partners = awi.my_partners[:self.n_partners]
        responses = dict()
        minprice = (
            awi.current_output_issues
            if awi.is_first_level
            else awi.current_input_issues
        )[UNIT_PRICE].min_value
        
        for partner, offer in zip(partners, offers):
            if offer[0] == offer[1] == 0:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue

            outcome = (offer[0], awi.current_step, offer[1] + minprice)

            if self.extra_checks:
                os = (
                    awi.current_input_outcome_space
                    if awi.is_first_level
                    else awi.current_output_outcome_space
                )
                assert (
                    outcome in os
                ), f"received {outcome} from {partner} ({offer}) in step {awi.current_step} for OS {os}\n{encoded=}"
            responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, outcome)

        return responses

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        if env._n_lines != self.n_lines:
            return False
        if env._n_suppliers != self.n_suppliers:
            return False
        if env._n_consumers != self.n_consumers:
            return False
        return True


@define(frozen=True)
class DictBetterObservationManager(BaseObservationManager):
    """
    An observation manager that utilizes a Dict observation space to easily and accurately capture many important aspects of the current state
    """
    extra_checks: bool = True
    n_prices: int = 2
    negotiation_limit: int = 40
    max_steps: int = 200
    max_disposal_cost: float = 0.3
    max_shortfall_penalty: float = 1.3
    max_production_cost: int = 20
    max_trading_price: int = 40
    n_partners: int = field(init=False)
    n_suppliers: int = field(init=False)
    n_consumers: int = field(init=False)
    max_quantity: int = field(init=False)
    n_lines: int = field(init=False)
    max_n_competitors: int = field(init=False)

    def _get_max_n_competitors(self):
        if isinstance(self.factory.n_agents_per_level, tuple):
            max_agents_per_level = self.factory.n_agents_per_level[-1]
        else:
            max_agents_per_level = self.factory.n_agents_per_level
        if isinstance(self.factory.n_competitors, tuple):
            max_n_competitors = self.factory.n_competitors[-1]
        else:
            max_n_competitors = self.factory.n_competitors
        return max(max_agents_per_level, max_n_competitors)

    def __attrs_post_init__(self):
        assert isinstance(self.factory, FixedPartnerNumbersOneShotFactory)
        object.__setattr__(self, "n_suppliers", self.factory.n_suppliers)
        object.__setattr__(self, "n_consumers", self.factory.n_consumers)
        object.__setattr__(self, "max_quantity", self.factory.n_lines)
        object.__setattr__(self, "n_lines", self.factory.n_lines)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)
        object.__setattr__(self, "max_n_competitors", self._get_max_n_competitors())

    def make_space(self) -> spaces.Space:
        """
        Creates the observation space.
        
        The observation space consists of:
            - The quantity and price (represented as high and low) for each incoming offer
            - The current needed quantity
            - The current round of negotations
            - The number of competitors (i.e. same-level factories)
            - The current simulation step
            - The curent disposal cost
            - The current shortfall penalty
            - The production cost
            - The current input trading price
            - The current output trading price
        """

        return spaces.Dict(
            {
                "offers": spaces.MultiDiscrete(
                    np.asarray([self.max_quantity + 1, self.n_prices] * self.n_partners).flatten()
                ),
                "needs": spaces.Discrete(self.max_quantity + 1),
                "round": spaces.Discrete(self.negotiation_limit + 1),
                "n_competitors": spaces.Discrete(self.max_n_competitors + 1),
                "simulation_step": spaces.Discrete(self.max_steps + 1),
                "disposal_cost": spaces.Box(low=0.0, high=self.max_disposal_cost),
                "shortfall_penalty": spaces.Box(low=0.0, high=self.max_shortfall_penalty),
                "production_cost": spaces.Discrete(self.max_production_cost + 1),
                "input_trading_price": spaces.Box(low=0.0, high=self.max_trading_price),
                "output_trading_price": spaces.Box(low=0.0, high=self.max_trading_price),
            }
        )
    
    def make_first_observation(self, awi: OneShotAWI):
        return self.encode(awi.state)

    def encode(self, state : OneShotState):
        obs = dict()

        partners = state.my_partners
        partner_index = dict(zip(partners, range(len(partners))))
        infos: list[NegotiationDetails] = [None] * len(partners)  # type: ignore
        negotiation_round = 0
        neg_relative_time = 0.0
        partners = list(
            itertools.chain(
                state.current_negotiation_details["buy"].items(),
                state.current_negotiation_details["sell"].items(),
            )
        )
        for partner, info in partners:
            infos[partner_index[partner]] = info
            negotiation_round = max(negotiation_round, info.nmi.state.step)
            neg_relative_time = max(neg_relative_time, info.nmi.state.relative_time)
        offers = [
            (0, 0)
            if info is None or info.nmi.state.current_offer is None  # type: ignore
            else (
                int(info.nmi.state.current_offer[QUANTITY]),  # type: ignore
                int(info.nmi.state.current_offer[UNIT_PRICE] - info.nmi.outcome_space.issues[UNIT_PRICE].min_value),  # type: ignore
            )
            for info in infos
        ]
        if self.extra_checks:
            assert (
                len(partners) == self.n_partners
            ), f"{len(partners)=} while {self.n_partners=}: {partners=}"
            assert (
                len(infos) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {infos=}"
            assert (
                len(offers) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {offers=}"

        obs["offers"] = np.asarray(offers).flatten()
        obs["needs"] = max(0, state.needed_sales if state.level == 0 else state.needed_supplies)
        obs["round"] = negotiation_round
        obs["n_competitors"] = state.n_competitors
        obs["simulation_step"] = state.current_step
        obs["disposal_cost"] = np.asarray([min(state.disposal_cost, self.max_disposal_cost)])
        obs["shortfall_penalty"] = np.asarray([min(state.shortfall_penalty, self.max_shortfall_penalty)])
        obs["production_cost"] = min(state.profile.cost, self.max_production_cost)
        obs["input_trading_price"] = np.asarray([min(state.trading_prices[state.my_input_product], self.max_trading_price)])
        obs["output_trading_price"] = np.asarray([min(state.trading_prices[state.my_output_product], self.max_trading_price)])

        return obs
    
    def get_offers(self, awi: OneShotAWI, encoded : dict[str, np.ndarray | int | float]) -> dict[str, SAOResponse]:
        offers = encoded["offers"].reshape((self.n_partners, 2))
        partners = awi.my_partners[:self.n_partners]
        responses = dict()
        minprice = (
            awi.current_output_issues
            if awi.is_first_level
            else awi.current_input_issues
        )[UNIT_PRICE].min_value
        
        for partner, offer in zip(partners, offers):
            if offer[0] == offer[1] == 0:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue

            outcome = (offer[0], awi.current_step, offer[1] + minprice)

            if self.extra_checks:
                os = (
                    awi.current_input_outcome_space
                    if awi.is_first_level
                    else awi.current_output_outcome_space
                )
                assert (
                    outcome in os
                ), f"received {outcome} from {partner} ({offer}) in step {awi.current_step} for OS {os}\n{encoded=}"
            responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, outcome)

        return responses
    

@define(frozen=True)
class FixedPartnerNumbersObservationManager_History(ObservationManager):
    n_bins: int = 10
    n_sigmas: int = 2
    extra_checks: bool = True
    n_prices: int = 2
    observation_history_length: int = 5
    contract_history_length: int = 5
    negotiation_limit : int = 20
    n_partners: int = field(init=False)
    n_suppliers: int = field(init=False)
    n_consumers: int = field(init=False)
    max_quantity: int = field(init=False)
    n_lines: int = field(init=False)
    observation_history: list = field(init=False)
    
    # def __init__(self, factory=ObservationManager.factory, extra_checks=self.extra_checks):
    #     self.observation_history=[0] * (((2 * 4) + 7) * 5)
    #     # self.observation_history=[0] * (((2 * self.n_partners) + 7) * self.observation_history_length)
        
    def __attrs_post_init__(self):
        assert isinstance(self.factory, FixedPartnerNumbersOneShotFactory)
        object.__setattr__(self, "n_suppliers", self.factory.n_suppliers)
        object.__setattr__(self, "n_consumers", self.factory.n_consumers)
        object.__setattr__(self, "max_quantity", self.factory.n_lines)
        object.__setattr__(self, "n_lines", self.factory.n_lines)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)
        object.__setattr__(self, "observation_history", [0] * (((2 * self.n_partners) + 7) * self.observation_history_length))

    def make_space(self) -> spaces.Space:
        """
        Creates the observation space
        
        The observation space consists of:
            - The quantity and price for each incoming offer
# New       - Last 5 observations recorded
            - The current needed quantity
            - The current round of negotations
            - The number of competitors
            - The current relative simulation time
            - The curent disposal cost (** how is this going to be represented? **)
            - The current shortfall penalty (** how is this going to be represented? **)
            - The relative price increase from input to output product
        """
# IDEA      - Last 5 signed contracts with each partner

        if isinstance(self.factory.n_agents_per_level, tuple):
            max_agents_per_level = self.factory.n_agents_per_level[-1]
        else:
            max_agents_per_level = self.factory.n_agents_per_level
        if isinstance(self.factory.n_competitors, tuple):
            max_n_competitors = self.factory.n_competitors[-1]
        else:
            max_n_competitors = self.factory.n_competitors
        return spaces.MultiDiscrete(
            np.asarray(
                (list(
                    itertools.chain(
                        [self.max_quantity + 1, self.n_prices] * self.n_partners
                    )
                )
                + [self.max_quantity + 1]
                + [self.negotiation_limit]
                + [max(max_agents_per_level, max_n_competitors)]
                + [self.n_bins + 1] * 4
                )
                * self.observation_history_length
                # + ([self.max_quantity + 1, 20, self.n_prices]* self.contract_history_length )* self.n_partners
            ).flatten()
        )
    
    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi.state)

    def encode(self, state: OneShotState) -> np.ndarray:
        """Encodes the state as an array"""
        partners = state.my_partners
        partner_index = dict(zip(partners, range(len(partners))))
        infos: list[NegotiationDetails] = [None] * len(partners)  # type: ignore
        negotiation_round = 0
        neg_relative_time = 0.0
        partners = list(
            itertools.chain(
                state.current_negotiation_details["buy"].items(),
                state.current_negotiation_details["sell"].items()
            )
        )
        for partner, info in partners:
            infos[partner_index[partner]] = info
            negotiation_round = max(negotiation_round, info.nmi.state.step)
            neg_relative_time = max(neg_relative_time, info.nmi.state.relative_time)
        offers = [
            (0, 0)
            if info is None or info.nmi.state.current_offer is None  # type: ignore
            else (
                int(info.nmi.state.current_offer[QUANTITY]),  # type: ignore
                int(info.nmi.state.current_offer[UNIT_PRICE] - info.nmi.outcome_space.issues[UNIT_PRICE].min_value)  # type: ignore
            )
            for info in infos
        ]
        if self.extra_checks:
            assert (
                len(partners) == self.n_partners
            ), f"{len(partners)=} while {self.n_partners=}: {partners=}"
            assert (
                len(infos) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {infos=}"
            assert (
                len(offers) == self.n_partners
            ), f"{len(infos)=} while {self.n_partners=}: {offers=}"

        # TODO add more state values here and remember to add corresponding limits in the make_space function
        def _normalize(x, mu, sigma, n_sigmas=self.n_sigmas):
            """
            Normalizes x between 0 and 1 given that it is sampled from a normal (mu, sigma).
            This is actually a very stupid way to do it.
            """
            mn = mu - n_sigmas * sigma
            mx = mu + n_sigmas * sigma
            if abs(mn - mx) < 1e-6:
                return 1
            return max(0, min(1, (x - mn) / (mx - mn)))

        extra = [
            max(0, state.needed_sales if state.level == 0 else state.needed_supplies),
            negotiation_round,
            state.n_competitors,
            int(state.relative_simulation_time * self.n_bins + 0.5),
            int(
                _normalize(
                    state.disposal_cost,
                    state.profile.disposal_cost_mean,
                    state.profile.disposal_cost_dev,
                )
                * self.n_bins
                + 0.5
            ),
            int(
                _normalize(
                    state.shortfall_penalty,
                    state.profile.shortfall_penalty_mean,
                    state.profile.shortfall_penalty_dev,
                )
                * self.n_bins
                + 0.5
            ),
            int(
                self.n_bins
                * max(
                    1,
                    (
                        state.trading_prices[state.my_output_product]
                        - state.trading_prices[state.my_input_product]
                    )
                    / state.trading_prices[state.my_output_product],
                )
                + 0.5
            )
        ]

        current_obs = np.asarray(offers).flatten().tolist() + extra
        object.__setattr__(self, "observation_history", self.observation_history[-len(current_obs):] + self.observation_history[:-len(current_obs)])
        object.__setattr__(self, "observation_history", current_obs + self.observation_history[len(current_obs):])
        
        v = np.asarray(self.observation_history)
        
        if self.extra_checks:
            space = self.make_space()
            assert space is not None and space.shape is not None
            exp = space.shape[0]
            assert (
                len(v) == exp
            ), f"{len(v)=}, {len(extra)=}, {len(offers)=}, {exp=}, {self.n_partners=}\n{state.current_negotiation_details=}"
            assert all(
                -1 < a < b for a, b in zip(v, space.nvec)  # type: ignore
            ), f"{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (state.exogenous_input_quantity , state.total_supplies , state.total_sales , state.exogenous_output_quantity) }"  # type: ignore
        
        return v

    def is_valid(self, env) -> bool:
        """Checks that it is OK to use this observation manager with a given `OneShotEnv`"""
        if env._n_lines != self.n_lines:
            return False
        if env._n_suppliers != self.n_suppliers:
            return False
        if env._n_consumers != self.n_consumers:
            return False
        return True