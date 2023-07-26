"""
A first attempt at reward shaping.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from attr import define, field
from gymnasium import spaces
import itertools
from negmas.helpers.strings import itertools

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import OneShotState

from scml.oneshot.rl.reward import RewardFunction

def get_current_negotiation_round(awi : OneShotAWI):
    negotiation_round = 0
    partners = itertools.chain(
        awi.state.current_negotiation_details["buy"].items(), 
        awi.state.current_negotiation_details["sell"].items(),
    )
    for _, info in partners:
        negotiation_round = max(negotiation_round, info.nmi.state.step)
    return negotiation_round

class ReducingNeedsReward(RewardFunction):
    def before_action(self, awi: OneShotAWI) -> Any:
        needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        return (awi.current_balance, needs)
    
    def __call__(self, awi: OneShotAWI, action: dict[str, tuple[int, int, int]], info: Any) -> float:
        old_balance, old_needs = info
        current_needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        
        # return abs(old_needs) - abs(current_needs) if get_current_negotiation_round(awi) != 0 else 0
        return (awi.current_balance - old_balance) + ((abs(old_needs) - abs(current_needs)) if get_current_negotiation_round(awi) != 0 else 0)
    
class QuantityBasedReward(RewardFunction):
    def before_action(self, awi: OneShotAWI) -> Any:
        needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        return needs
    
    def __call__(self, awi: OneShotAWI, action: dict[str, tuple[int, int, int]], info: Any) -> float:
        needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        return abs(info) - abs(needs) if get_current_negotiation_round(awi) != 0 else 0

class TestReducingNeedsReward(RewardFunction):
    def before_action(self, awi: OneShotAWI) -> Any:
        needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        return needs

    def __call__(self, awi: OneShotAWI, action: dict[str, tuple[int, int, int]], info: Any) -> float:
        current_needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        return info - current_needs