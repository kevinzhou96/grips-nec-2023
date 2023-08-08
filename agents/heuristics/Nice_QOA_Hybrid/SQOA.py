# Modified form of QOA to be 'smart'. Changes from QOA are
## Takes into account level of agent
### If level 0, be more desparate to unload supply by always offering best
### price for opponent

### If level 1, less desparate, so we keep default QOA strategy

import math
import matplotlib.pyplot as plt
from collections import defaultdict
from negmas import ResponseType, MechanismState
from negmas.outcomes import Outcome
from negmas.sao import SAOResponse, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent

__all__ = ["SmartQuantityOrientedAgent"]

class SmartQuantityOrientedAgent(OneShotAgent):
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
        if self.awi.level == 0: # We are L0, so be desparate
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
        
        else: # We are not L0, so don't be desparate, go for default QOA strat
            p = 17
            step = state.step
            if (
                    step <= p
                    ):  # Insists on the best offer for itself for a given amount of time "p"
                return self.best_offer(negotiator_id, step)
            else:  # tries to settle using the best offer for its partners
                return self.quick_offer(negotiator_id, step)

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

    def best_offer(self, negotiator_id, step):
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
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def quick_offer(self, negotiator_id, step):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        if step <= 3:
            if my_needs <= 5:
                offer[QUANTITY] = my_needs
            else:
                offer[QUANTITY] = math.ceil(my_needs / 2)  # First offer: splits demand.
        else:
            offer[QUANTITY] = my_needs  # Offers exactly what the agent needs

        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[
                UNIT_PRICE
            ] = unit_price_issue.min_value  # Offers the best value FOR THE BUYER!!!
        else:
            offer[
                UNIT_PRICE
            ] = unit_price_issue.max_value  # Offers the best value FOR THE SELLER!!!
        return tuple(offer)