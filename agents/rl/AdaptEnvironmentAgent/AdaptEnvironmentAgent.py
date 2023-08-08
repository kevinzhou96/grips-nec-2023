

class AdoptEnvironmentAgent(OneShotSyncAgent):
    """Based on OneShotSyncAgent"""

    def init(self):
        """Initializes some values needed for the negotiations"""
        self.is_seller = True if self.awi.profile.level == 0 else False
        if self.awi.profile.level == 0:
            self.level = 0
        else:
            self.level = 1

        self.compromise_point = 0.7
        self.pursue_points = 0.3
        self.max_compromise_need = 5
        self.final_needs_list = []
        self.permit_needs = 0


    def before_step(self):
        self.secured = 0
        self.first_needs = self.awi.current_exogenous_input_quantity + self.awi.current_exogenous_output_quantity

        #renew
        if len(self.final_needs_list) >= 10:
            ave = sum(x for x in self.final_needs_list if x > 0) / len(self.final_needs_list)
            if ave > self.compromise_point:
                self.permit_needs = min([self.permit_needs + 1, self.max_compromise_need])
            elif ave < self.pursue_points:
                self.permit_needs = max([self.permit_needs - 1, 0])
            self.final_needs_list = []
        self.current_permit_needs = self.permit_needs


    def step(self):
        self.final_needs_list.append(self.first_needs - self.secured)


    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]


    def get_proposals(self, partners, needs):
        step = self.awi.current_step
        level = self.level
        n_of_partners = len(partners)

        offer_price = self.bullish_price()
        if n_of_partners == 1:
            propose_quantity = needs
        elif needs <= 9:
            propose_quantity = math.ceil(needs/2)
        else:
            propose_quantity = math.ceil(needs/2) + 1

        partners_quantities = {partner: propose_quantity for partner in partners}

        proposal_dict = {
            partner: self.get_outcome(offer_price, quantity, step)
            for partner, quantity in partners_quantities.items()
        }
        return proposal_dict


    def first_proposals(self) -> dict[str, Outcome]:
        partners = set(self.partners())
        needs = self.needed_quantity()
        return self.get_proposals(partners, needs)


    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:

        current_step = list(states.values())[0].step
        needs = self.needed_quantity()
        all_partners = set(offers.keys())

        if needs <= 0:
            return {
                agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                for agent in all_partners
            }

        #decide acceptance strategy
        else:
            m = len(all_partners)
            up_quantity_list = []
            for i in range(m):
                up_quantity_list.append([list(offers.values())[i][UNIT_PRICE], list(offers.values())[i][QUANTITY]])
            sum_quantity = 0
            sum_price = 0
            compromised_false = 0
            decide_choice_number = 0


            for choice_number in range(pow(2,m)):
                false = 0
                current_price = 0
                current_quantity = 0
                for i in range(m):
                    if (choice_number // pow(2,i)) % 2 == 1:
                        if (self.level == 1) and (up_quantity_list[i][0] != self.bullish_price()):
                            false += up_quantity_list[i][1]
                        current_quantity += up_quantity_list[i][1]
                        current_price += up_quantity_list[i][0] * up_quantity_list[i][1]

                if ((false <= self.current_permit_needs) or (current_step >= 18)) and (current_quantity <= needs) and (current_quantity > sum_quantity):
                    sum_quantity = current_quantity
                    sum_price = current_price
                    decide_choice_number = choice_number
                    compromised_false = false
                elif ((false <= self.current_permit_needs) or (current_step >= 18)) and (current_quantity <= needs) and (current_quantity == sum_quantity):
                    previous_abs = abs(sum_price - self.bullish_price() * sum_quantity)
                    current_abs = abs(current_price - self.bullish_price() * current_quantity)
                    if current_abs < previous_abs:
                        sum_price = current_price
                        decide_choice_number = choice_number
                        compromised_false = false

            self.current_permit_needs -= compromised_false

            acceptable_partners = set()
            keys = list(offers.keys())
            for i in range(m):
                if (decide_choice_number // pow(2,i)) % 2 == 1:
                    acceptable_partners.add(keys[i])

            acceptance_dict: dict[str, SAOResponse] = {
                agent: SAOResponse(ResponseType.ACCEPT_OFFER, None)
                for agent in acceptable_partners
            }

            #decide reject or end to others
            if (needs == sum_quantity) or (len(all_partners - acceptable_partners) == 0):
                end_partners = all_partners - acceptable_partners
                end_negotiation_dict: dict[str, SAOResponse] ={
                    agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for agent in end_partners
                }
                result = acceptance_dict | end_negotiation_dict
            else:
                reject_partners = all_partners - acceptable_partners
                agent_proposals = self.get_proposals(reject_partners, needs - sum_quantity)
                reject_dict : dict[str, SAOResponse] ={
                    agent: SAOResponse(ResponseType.REJECT_OFFER, proposal)
                    for agent, proposal in agent_proposals.items()
                }
                result = acceptance_dict | reject_dict
            return result

    def partners(self) -> list[str]:
        return self.awi.my_consumers if self.is_seller else self.awi.my_suppliers

    def needed_quantity(self):
        return self.first_needs - self.secured

    def get_issues(self) -> list[Issue]:
        if self.is_seller:
            return self.awi.current_output_issues
        else:
            return self.awi.current_input_issues

    def get_price_range(self) -> tuple[int, int]:
        price_issue = self.get_issues()[UNIT_PRICE]
        return price_issue.min_value, price_issue.max_value

    def bullish_price(self) -> int:
        mn, mx = self.get_price_range()
        self.awi.logdebug_agent(f"(mn,mx)={(mn, mx)}")
        if self.is_seller:
            return mx
        else:
            return mn

    def bearish_price(self) -> int:
        mn, mx = self.get_price_range()
        self.awi.logdebug_agent(f"(mn,mx)={(mn, mx)}")
        if self.is_seller:
            return mn
        else:
            return mx

    def get_outcome(self, unit_price: int, quantity: int, time: int) -> Outcome:
        offer = [0, 0, 0]
        offer[UNIT_PRICE] = unit_price
        offer[QUANTITY] = quantity
        offer[TIME] = time
        return tuple(offer)
