"""You need two files Q_propose_matrix.csv, Q_respond_matrix.csv. """

class AgentByQValue0726(OneShotSyncAgent):
    """Based on OneShotAgent"""

    def init(self):
        #print("init OK? (QL)")
        """Initializes some values needed for the negotiations"""
        self.is_seller = True if self.awi.profile.level == 0 else False
        if self.awi.profile.level == 0:
            self.level = 0
        else:
            self.level = 1
        self.Q_value_matrix = np.loadtxt("Q_value_matrix.csv", delimiter=",")
        if use_data_of_needs_of_days:
            self.final_needs_of_byQvalueAgent = load_data_from_csv("final_needs_of_byQvalueAgent.csv")

    def before_step(self):  # Resets counts
        #print("before_step OK? (QL)")
        self.secured = 0
        self.first_needs = self.awi.current_exogenous_input_quantity + self.awi.current_exogenous_output_quantity
        self.contracts_list = []

    if True:
        def step(self):
            print("final needs by Q-value= ", self.first_needs - self.secured)
            if use_data_of_needs_of_days:
                #global final_needs_of_byQvalueAgent
                self.final_needs_of_byQvalueAgent.append(self.first_needs - self.secured)
                save_to_csv(self.final_needs_of_byQvalueAgent, "final_needs_of_byQvalueAgent.csv")
            if self.first_needs - self.secured <= -10:
                print("self.first_needs, self.secured = ", self.first_needs, self.secured)
#                for contract in self.contracts_list:
#                    print("contract = ", contract)

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]
#        self.contracts_list.append(contract)
#        self.contracts_list.append(self.current_state_act)
#        self.contracts_list.append([[self.current_state_act, contract]])

    if False:
        def on_negotiation_failure(self, contract, mechanism, a, b):
            a = 0

    if False:
        def _is_selling(self, ami):
            return ami.annotation["product"] == self.awi.my_output_product


    def get_proposals(self, partners, needs):
        #print("get_proposalsが呼び出された")
        step = self.awi.current_step
        level = self.level
        n_of_partners = len(partners)

        value_state = min(step, 20) + 21 * (max(needs,-2)+2) + 273 * level + 546 * (n_of_partners - 1) #for unit_price and quantity
        value_act = self.get_random_max_col_index(self.Q_value_matrix, value_state)

        if True:
            if np.all(self.Q_value_matrix[value_state] == 0):
                print("not learn ", [step, needs, level, n_of_partners])

        if treat_level:
            if value_act < len(choice_of_quantity):
                offer_price = self.bearish_price()
                propose_quantity = choice_of_quantity[value_act](needs, n_of_partners)
            else:
                offer_price = self.bullish_price()
                propose_quantity = choice_of_quantity[value_act - len(choice_of_quantity)](needs, n_of_partners)
            propose_quantity = choice_of_quantity[value_act % len(choice_of_quantity)](needs, n_of_partners)
        else:
            if self.level == 0:
                offer_price = self.bearish_price()
            else:
                offer_price = self.bullish_price()
            propose_quantity = choice_of_quantity[value_act](needs, n_of_partners)

        #print("offer_price, propose_quantityが計算できた")
        partners_quantities = {partner: propose_quantity for partner in partners}

        proposal_dict = {
            partner: self.get_outcome(offer_price, quantity, step)
            for partner, quantity in partners_quantities.items()
        }
        #print("proposal_dictを作れた")
        #print("proposal_dict =", proposal_dict)
        return proposal_dict


    def first_proposals(self) -> dict[str, Outcome]:
        partners = set(self.partners())
        needs = self.needed_quantity()
        #print("first_proposalsが呼び出された")
        return self.get_proposals(partners, needs)

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        #print("counter_allが呼び出された")
        #print("received offers = ", offers)
        #print("received states = ", states)

        current_step = list(states.values())[0].step
        #print("current_step =", current_step)
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
            decide_choice_number = 0

            if True:
                for choice_number in range(pow(2,m)):
                    false = 0
                    current_price = 0
                    current_quantity = 0
                    for i in range(m):
                        if (choice_number // pow(2,i)) % 2 == 1:
                            if (self.level == 1) and (up_quantity_list[i][0] != self.bullish_price()):
                                false = 1
                            current_quantity += up_quantity_list[i][1]
                            current_price += up_quantity_list[i][0] * up_quantity_list[i][1]
                    if ((false == 0) or (current_step >= 19)) and (current_quantity <= needs) and (current_quantity > sum_quantity):
                        sum_quantity = current_quantity
                        sum_price = current_price
                        decide_choice_number = choice_number
                    elif ((false == 0) or (current_step >= 19)) and (current_quantity <= needs) and (current_quantity == sum_quantity):
                        if False:
                            if sum_quantity != 0:
                                print("sum_price, sum_quantity =", sum_price, sum_quantity)
                                print("current_price, current_quantity =", current_price, current_quantity)
                                print("needs, level =", needs, self.level)
                                print("ideal_price =", self.bullish_price() * sum_quantity)
                                a = sum_price - self.bullish_price() * sum_quantity
                                b = current_price - self.bullish_price() * current_quantity
                                print("a=", a)
                                print("b=", b)
                        previous_abs = abs(sum_price - self.bullish_price() * sum_quantity)
                        current_abs = abs(current_price - self.bullish_price() * current_quantity)
                        if current_abs < previous_abs:
                            sum_price = current_price
                            decide_choice_number = choice_number
            acceptable_partners = set()
            keys = list(offers.keys())
            for i in range(m):
                if (decide_choice_number // pow(2,i)) % 2 == 1:
                    acceptable_partners.add(keys[i])

            acceptance_dict: dict[str, SAOResponse] = {
                agent: SAOResponse(ResponseType.ACCEPT_OFFER, None)
                for agent in acceptable_partners
            }
            if False:
                if len(acceptable_partners) > 0:
                    print("received offers = ", offers)
                    print("needs, level, accept =", needs, self.level, acceptable_partners)

            #decide reject or end to others
            if (needs == sum_quantity) or (len(all_partners - acceptable_partners) == 0):
                end_partners = all_partners - acceptable_partners
                end_negotiation_dict: dict[str, SAOResponse] ={
                    agent: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    for agent in end_partners
                }
                result = acceptance_dict | end_negotiation_dict
            else:
                if needs - sum_quantity <= 0:
                    print("アウト　needs, sum_quantity =", needs, sum_quantity)
                reject_partners = all_partners - acceptable_partners
                agent_proposals = self.get_proposals(reject_partners, needs - sum_quantity)
                reject_dict : dict[str, SAOResponse] ={
                    agent: SAOResponse(ResponseType.REJECT_OFFER, proposal)
                    for agent, proposal in agent_proposals.items()
                }
                result = acceptance_dict | reject_dict
            #print("counter_allを作れた")
            return result

    def partners(self) -> list[str]:
        return self.awi.my_consumers if self.is_seller else self.awi.my_suppliers

    def needed_quantity(self):
        return self.first_needs - self.secured

    def get_issues(self) -> list[Issue]:
        # self.awi.logdebug_agent(f"i: {self.awi.current_input_issues}, o: {self.awi.current_output_issues}")
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

    def find_max_column(self, data, row_index):
        row = data[row_index]
        #max_value = np.max(row)
        max_column = np.argmax(row)
        return max_column

    def get_random_max_col_index(self, matrix, row):
        max_value = np.max(matrix[row])
        max_indices = np.where(matrix[row] == max_value)[0]
        random_index = random.choice(max_indices)
        return random_index
