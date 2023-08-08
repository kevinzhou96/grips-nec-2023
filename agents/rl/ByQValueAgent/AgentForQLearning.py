"""You need file Q_value_matrix.csv """

class AgentForQLearning0726(OneShotSyncAgent):
    """Based on OneShotSyncAgent"""

    def init(self):
        #print("init OK? (QL)")
        """Initializes some values needed for the negotiations"""
        self.is_seller = True if self.awi.profile.level == 0 else False
        if self.awi.profile.level == 0:
            self.level = 0
        else:
            self.level = 1

        global Counter
        Counter += 1
        print("Counter = ", Counter)
        global save_counter
        global dawnload_counter
        global Q_value_matrix
        if Counter % save_counter == 0:
            np.savetxt("Q_value_matrix.csv", Q_value_matrix, delimiter=",")
        if False:
            if Counter % dawnload_counter == 0:
                files.download("Q_value_matrix.csv")
        print("Sum of Q_value_matrix = ", np.sum(Q_value_matrix))

    def before_step(self):  # Resets counts
        #print("before_step OK? (QL)")
        self.secured = 0
        self.first_needs = self.awi.current_exogenous_input_quantity + self.awi.current_exogenous_output_quantity

        if self.awi.level == 0:
            self.output = [True]
        elif self.awi.level == 1:
            self.output = [False]

        self.state_act = "initial state"
        self.Q_value_matrix = self.call_Q_value_matrix()
        self.already_contracts = ()
        self.prospective_contracts = ()
        self.previous_contracts = ()



    def step(self):
        #print("final needs = ", self.first_needs - self.secured)
        global Q_value_matrix
        if self.state_act != "initial state":
            new_state = [20, self.first_needs - self.secured, self.level, 0, 0, 0]
            self.update_Q_matrix(self.state_act, new_state)
        self.save_Q_value_matrix()


    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

        quantity = contract.agreement["quantity"]
        time = contract.agreement["time"]
        unit_price = contract.agreement["unit_price"]
        self.already_contracts += (self.get_outcome(unit_price, quantity, time),)
        self.prospective_contracts = self.already_contracts

    def get_proposals(self, partners, needs):
        #print("get_proposalsが呼び出された")
        step = self.awi.current_step
        level = self.level
        n_of_partners = len(partners)

        global Q_value_matrix
        if self.state_act != "initial state":
            new_state = [int(step), needs, level, n_of_partners, 0, 0]
            self.update_Q_matrix(self.state_act, new_state)
            self.state_act = new_state
            self.previous_contracts = self.prospective_contracts
        else:
            self.state_act = [int(step), needs, level, n_of_partners, 0, 0]

        random_value = np.random.uniform(0, 1)
        global epsilon0
        if random_value < epsilon0:        #whether action is random or non random
            value_state = min(step, 20) + 21 * (max(needs,-2)+2) + 273 * level + 546 * (n_of_partners - 1) #for unit_price and quantity
            value_act = self.get_random_max_col_index(self.Q_value_matrix, value_state)

            if treat_level:
                if value_act < len(choice_of_quantity):
                    offer_price = self.bearish_price()
                    propose_quantity = choice_of_quantity[value_act](needs, n_of_partners)
                    self.state_act[5] = 0
                    self.state_act[4] = value_act
                else:
                    offer_price = self.bullish_price()
                    propose_quantity = choice_of_quantity[value_act - len(choice_of_quantity)](needs, n_of_partners)
                    self.state_act[5] = 1
                    self.state_act[4] = value_act - len(choice_of_quantity)
            else:
                if self.level == 0:
                    offer_price = self.bearish_price()
                else:
                    offer_price = self.bullish_price()
                propose_quantity = choice_of_quantity[value_act](needs, n_of_partners)
                self.state_act[4] = value_act

        else:
            if False:
                selected_number0 = random.choice(range(len(choice_of_quantity)))
                propose_quantity = choice_of_quantity[selected_number0](needs, n_of_partners)
                self.state_act[4] = selected_number0
                if treat_level:
                    selected_number1 = random.choice(range(2))
                    if selected_number1 == 0:
                        offer_price = self.bearish_price()
                        self.state_act[5] = 0
                    else:
                        offer_price = self.bullish_price()
                        self.state_act[5] = 1
                else:
                    if level == 0:
                        offer_price = self.bearish_price()
                    else:
                        offer_price = self.bullish_price()

            else:
                value_state = min(step, 20) + 21 * (max(needs,-2)+2) + 273 * level + 546 * (n_of_partners - 1) #for unit_price and quantity
                col = self.get_random_column_index(self.Q_value_matrix, value_state, -500)
                propose_quantity = choice_of_quantity[col % len(choice_of_quantity)](needs, n_of_partners)
                self.state_act[4] = col % len(choice_of_quantity)
                if treat_level:
                    if col < len(choice_of_quantity):
                        offer_price = self.bearish_price()
                        self.state_act[5] = 0
                    else:
                        offer_price = self.bullish_price()
                        self.state_act[5] = 1
                else:
                    if level == 0:
                        offer_price = self.bearish_price()
                    else:
                        offer_price = self.bullish_price()

        partners_quantities = {partner: propose_quantity for partner in partners}
        proposal_dict = {
            partner: self.get_outcome(offer_price, quantity, step)
            for partner, quantity in partners_quantities.items()
        }
        return proposal_dict

    def first_proposals(self) -> dict[str, Outcome]:
        partners = set(self.partners())
        needs = self.needed_quantity()
        if len(partners) == 0:
            print("初手から誰もいない")
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
                    self.prospective_contracts += (offers[keys[i]],)
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

    def update_Q_matrix(self, state_act, new_state):
        global Q_value_matrix
        global learning_rate
        global discount_factor
        row = self.translate_state(state_act)
        col = self.translate_act(state_act)
        q0 = self.Q_value_matrix[row, col]

        prospective_ufun = self.ufun.from_offers(self.prospective_contracts, outputs=tuple(self.output * (len(self.prospective_contracts))),)
        previous_ufun = self.ufun.from_offers(self.previous_contracts, outputs=tuple(self.output * (len(self.previous_contracts))),)
        reward = prospective_ufun - previous_ufun

        if new_state[0] != 20:
            q1 = np.amax(self.Q_value_matrix[self.translate_state(new_state)])
        else:
            q1 = 0
        result = q0 + learning_rate * (reward + discount_factor * q1 - q0)
        if False:
            print("state_act, new_state =", state_act, new_state)
            print("q0, reward, q1, result =", q0, reward, q1, result)
            print("  ")
        self.Q_value_matrix[self.translate_state(state_act), self.translate_act(state_act)] = result

    def translate_state(self, state_act):
        step = state_act[0]
        Needs = state_act[1]
        level = state_act[2]
        n_of_partners = state_act[3]

        Needs = max(Needs, -2) + 2
        step = min(step, 20)
        n_of_partners -= 1

        row = step + 21 * Needs + 273 * level + 546 * n_of_partners
        return row

    def translate_act(self, state_act):
        selected_choice_of_quantity = state_act[4]
        unit_price = state_act[5]

        if treat_level:
            col = selected_choice_of_quantity + len(choice_of_quantity) * unit_price
        else:
            col = selected_choice_of_quantity
        return col

    def find_max_column(self, data, row_index):
        row = data[row_index]
        max_column = np.argmax(row)
        return max_column

    def get_random_max_col_index(self, matrix, row):
        max_value = np.max(matrix[row])
        max_indices = np.where(matrix[row] == max_value)[0]
        random_index = random.choice(max_indices)
        return random_index

    def call_Q_value_matrix(self):
        global Q_value_matrix
        return Q_value_matrix

    def save_Q_value_matrix(self):
        global Q_value_matrix
        Q_value_matrix = self.Q_value_matrix

    def get_random_column_index(self, data, row_number, a):
        row = data[row_number]
        indices = np.where(row > a)[0]
        if indices.size == 0:
            return None
        return random.choice(indices)
