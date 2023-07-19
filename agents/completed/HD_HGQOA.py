# Attempt at outperforming QOA overall.
## Level 0 strategy is similar to QOA with a slight change.
### When opponent offers us a quantity that is more than our needs,
### still reject their offer, but record the offered quantity.
### On our proposal, offer the opponent the minimum of our current needs and
### half their previous offer.

## Level 1 strategy is a half-greedy strategy that rejects partner offers
## unless the price is our best price and the quantity is <= our needs.
## Otherwise, record the quantity offered by the opponent and counteroffer
## for our best price and the minimum of our needs and half of what they
## last requested.

import math
import matplotlib.pyplot as plt
from collections import defaultdict
from negmas import ResponseType, MechanismState
from negmas.outcomes import Outcome
from negmas.sao import SAOResponse, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent

__all__ = ["HalfDesparateHalfGreedyQOA"]


class HalfDesparateHalfGreedyQOA(OneShotAgent):
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
        # Added initialization of happy quantity measurements
        self._happy_opp_quantity = defaultdict(int)

    def on_negotiation_success(self, contract, mechanism):
        if self.awi.level == 1:
            self.secured += contract.agreement["quantity"]

    def on_negotiation_failure(self, contract, mechanism, a, b):
        if self.awi.level == 1:
            self.rejection = self.rejection + 1  # Tracks the number of rejections

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def propose(self, negotiator_id: str, state):
        if self.awi.level == 1: # HGQOA behavior
            p = 17
            step = state.step
            if (
                step <= p
            ):  # Insists on the best offer for itself for a given amount of time "p"
            
                return self.best_offer(negotiator_id, step)
            else:  # tries to settle using the best offer for its partners
                return self.quick_offer(negotiator_id, step)
        else: # Half Desperate QOA behavior
            my_needs = self._needed(negotiator_id)
            if my_needs <= 0:
                return None
            ami = self.get_nmi(negotiator_id)
            if not ami:
                return None
            quantity_issue = ami.issues[QUANTITY]
            unit_price_issue = ami.issues[UNIT_PRICE]
            offer = [-1] * 3
            # First, check opponent's last proposed quantity
            if self._is_selling(ami):
                partner = ami.annotation["buyer"]
                if partner in self._happy_opp_quantity:
                    offer[QUANTITY] = max(min(math.ceil(float(self._happy_opp_quantity[partner]) / 2), my_needs),1)
                    offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].min_value
                    offer[TIME] = self.awi.current_step
                    return tuple(offer)
                else: # This should only be called when we're making the first proposals for a day
                    if my_needs >= 5:
                        offer[QUANTITY] = max(math.ceil(float(my_needs) / 2), 1)
                        offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].max_value
                        offer[TIME] = self.awi.current_step
                        return tuple(offer)
                    else:
                        offer[QUANTITY] = my_needs
                        offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].max_value
                        offer[TIME] = self.awi.current_step
                        return tuple(offer)
                    
            else:
                partner = ami.annotation["seller"]
                if partner in self._happy_opp_quantity:
                    offer[QUANTITY] = max(min(math.ceil(float(self._happy_opp_quantity[partner]) / 2), my_needs),1)
                    offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].max_value
                    offer[TIME] = self.awi.current_step
                    return tuple(offer)
                else: # This should only be called when we're making the first proposals for a day
                    if my_needs >= 5:
                        offer[QUANTITY] = max(math.ceil(float(my_needs) / 2), 1)
                        offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].min_value
                        offer[TIME] = self.awi.current_step
                        return tuple(offer)
                    else:
                        offer[QUANTITY] = my_needs
                        offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].min_value
                        offer[TIME] = self.awi.current_step
                        return tuple(offer)
                    

    def respond(self, negotiator_id, state, offer, source=""):
        if self.awi.level == 0: # Half Smart QOA behavior
            ami = self.get_nmi(negotiator_id)
            step = state.step
            my_needs = self._needed(negotiator_id)
            if my_needs <= 0:
                return ResponseType.END_NEGOTIATION
            else: # Desparate behavior, accepts when offered quantity is below
                  # needed level. Otherwise, records opponents offer for counter
                if (offer[QUANTITY] <= my_needs):
                    return ResponseType.ACCEPT_OFFER
                elif ( # Change in late-stage negotiation behavior. If my opponent has the last move, accept if it takes
                    # me closer to goal quantity
                    abs(offer[QUANTITY] - my_needs) < my_needs and step >= 18
                ):
                      # At the very last round, settle for an offer that minimizes the losses
                        return ResponseType.ACCEPT_OFFER
                elif self._is_selling(ami):
                    partner = ami.annotation["buyer"]
                    self._happy_opp_quantity[partner] = offer[QUANTITY]
                    return ResponseType.REJECT_OFFER
                else:
                    partner = ami.annotation["seller"]
                    self._happy_opp_quantity[partner] = offer[QUANTITY]
                    return ResponseType.REJECT_OFFER
                
        else: # HGQOA behavior
            ami = self.get_nmi(negotiator_id)
            step = state.step
            my_needs = self._needed(negotiator_id)
            if my_needs <= 0:
                return ResponseType.END_NEGOTIATION
            else: # Change here to reject offer when <= my_needs, and record happy quantity levels
                if ( # Change in late-stage negotiation behavior. If my opponent has the last move, accept if it takes
                    # me closer to goal quantity
                    abs(offer[QUANTITY] - my_needs) < my_needs and step >= 18
                ):  # At the very last round, settle for an offer that minimizes the losses
                    return ResponseType.ACCEPT_OFFER
                elif (offer[QUANTITY] <= my_needs):
                    if self._is_selling(ami):
                        if offer[UNIT_PRICE] == ami.issues[UNIT_PRICE].max_value: 
                            return ResponseType.ACCEPT_OFFER # Sell if they offer my best price for acceptable quantity
                        else:
                            partner = ami.annotation["buyer"]
                            self._happy_opp_quantity[partner] = offer[QUANTITY]
                            return ResponseType.REJECT_OFFER
                    else:
                        if offer[UNIT_PRICE] == ami.issues[UNIT_PRICE].min_value:
                            return ResponseType.ACCEPT_OFFER # Buy if they offer my best price for acceptable quantity
                        else:
                            partner = ami.annotation["seller"]
                            self._happy_opp_quantity[partner] = offer[QUANTITY]
                            return ResponseType.REJECT_OFFER
                return ResponseType.REJECT_OFFER

    def best_offer(self, negotiator_id, step): # Function solely for HGQOA
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        # First, check if opponent has proposed an acceptable quantity
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            if partner in self._happy_opp_quantity:
                if self._happy_opp_quantity[partner] >= 5:
                    offer[QUANTITY] = min(math.ceil(float(self._happy_opp_quantity[partner]) / 2), my_needs)
                    offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].max_value
                    offer[TIME] = self.awi.current_step
                    return tuple(offer)
                else:
                    offer[QUANTITY] = min(self._happy_opp_quantity[partner], my_needs)
                    offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].max_value
                    offer[TIME] = self.awi.current_step
                    return tuple(offer)
                
        else:
            partner = ami.annotation["seller"]
            if partner in self._happy_opp_quantity:
                if self._happy_opp_quantity[partner] >= 5:
                    offer[QUANTITY] = min(math.ceil(float(self._happy_opp_quantity[partner]) / 2), my_needs)
                    offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].min_value
                    offer[TIME] = self.awi.current_step
                    return tuple(offer)
                else:
                    offer[QUANTITY] = min(self._happy_opp_quantity[partner], my_needs)
                    offer[UNIT_PRICE] = ami.issues[UNIT_PRICE].min_value
                    offer[TIME] = self.awi.current_step
                    return tuple(offer)
        # If we don't return in the above conditional statements, then we have yet to receive an
        # acceptable quantity offer from our negotiating partner. Thus, we go to our default "best" offer.
        # The above two added conditional sections are a bit greedier than QOA in that they try to
        # ask for better prices rather than just auto-accepting contracts with quantities that are
        # less than or equal to their needed amounts.
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

    def quick_offer(self, negotiator_id, step): # Function solely for HGQOA
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