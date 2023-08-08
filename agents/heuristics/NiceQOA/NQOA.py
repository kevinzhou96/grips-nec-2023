# Modified form of QOA to be 'nice'. Changes are
## Always proposes price that's best for the opponent
### This means we no longer need the best_offer and quick_offer functions


import math
import matplotlib.pyplot as plt
from collections import defaultdict
from negmas import ResponseType, MechanismState
from negmas.outcomes import Outcome
from negmas.sao import SAOResponse, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent

__all__ = ["NiceQuantityOrientedAgent"]

class NiceQuantityOrientedAgent(OneShotAgent):
    """Based on OneShotAgent"""

    def init(self):
        """Initializes some values needed for the negotiations"""
        if self.awi.level == 0:  # L0 agent
            self.n_partners = len(
                self.awi.all_consumers[1]
            )  # determines the number of l1 agents
        else:  # L1, Start the negotiations
            self.n_partners = len(
                self.awi.all_consumers[0]
            )  # determines the number of l0 agents

    def before_step(self):  # Resets counts
        self.secured = 0
        self.rejection = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def on_negotiation_failure(self, contract, mechanism, a, b):
        self.rejection = self.rejection + 1  # Tracks the numbers of rejections

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def propose(self, negotiator_id: str, state):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        if my_needs <= 5:
            offer[QUANTITY] = my_needs
        else:
            offer[QUANTITY] = math.ceil(my_needs / 2)  # splits demand.

        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.min_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.max_value
        return tuple(offer)

    def respond(self, negotiator_id, state, offer, source=""):
        ami = self.get_nmi(negotiator_id)
        step = state.step
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        else:
            if offer[QUANTITY] == my_needs:  # Best possible outcome
                return ResponseType.ACCEPT_OFFER
            elif offer[QUANTITY] < my_needs:
                return (
                    ResponseType.ACCEPT_OFFER
                )  # Also accepts if the quantity is lower than needed
            elif (
                abs(offer[QUANTITY] - my_needs) < my_needs and step >= 18
            ):  # At the last rounds, might settle for a offer that minimizes the losses
                return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER