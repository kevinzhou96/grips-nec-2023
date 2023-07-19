class AgentForQLearning0717(OneShotAgent):
    """Based on OneShotAgent"""

    def init(self):
        """Initializes some values needed for the negotiations"""
        global Counter
        Counter += 1
        print("Counter = ", Counter)
        global save_counter
        global Q_propose_matrix
        global Q_respond_matrix
        if Counter % save_counter == 0:
            np.savetxt("Q_propose_matrix.csv", Q_propose_matrix, delimiter=",")
            np.savetxt("Q_respond_matrix.csv", Q_respond_matrix, delimiter=",")
        print("Sum of Q_propose_matrix = ", np.sum(Q_propose_matrix))
        print("Sum of Q_respond_matrix = ", np.sum(Q_respond_matrix))

    def before_step(self):  # Resets counts
        self.secured = 0
        self.state_act = []
        self.first_needs = self.awi.current_exogenous_input_quantity + self.awi.current_exogenous_output_quantity
        self.name_list = []
        self.Q_propose_matrix = self.call_Q_propose_matrix()
        self.Q_respond_matrix = self.call_Q_respond_matrix()

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]
        my_id = self.awi.agent.id
        if type(contract) == list:
            if my_id == contract[1]:
                negotiator_id = contract[0]
            else:
                negotiator_id = contract[1]
        else:
            negotiator_id = (
                contract.partners[0]
                if my_id == contract.partners[1]
                else contract.partners[1]
            )
        opponent_number = self.name_list.index(negotiator_id)
        state_act_letter = self.state_act[opponent_number]
        state_act_letter[1] = self.first_needs - self.secured + contract.agreement["quantity"]
        new_state = [20, self.first_needs - self.secured, 0, 0, 0, self.awi.level]
        self.update_Q_matrix(state_act_letter, new_state)

    def on_negotiation_failure(self, contract, mechanism, a, b):
        my_id = self.awi.agent.id
        if type(contract) == list:
            if my_id == contract[1]:
                negotiator_id = contract[0]
            else:
                negotiator_id = contract[1]
        else:
            negotiator_id = (
                contract.partners[0]
                if my_id == contract.partners[1]
                else contract.partners[1]
            )
        opponent_number = self.name_list.index(negotiator_id)
        state_act_letter = self.state_act[opponent_number]
        new_state = [20, state_act_letter[1], 0, 0, 0, self.awi.level]
        self.update_Q_matrix(state_act_letter, new_state)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def step(self):
        global Q_propose_matrix
        global Q_respond_matrix
        for state_act_letter in self.state_act:
            if state_act_letter[4] != 0:
                new_state = [20, state_act_letter[1], 0, 0, 0, self.awi.level]
                self.update_Q_matrix(state_act_letter, new_state)
        self.save_Q_propose_matrix()
        self.save_Q_respond_matrix()

    def propose(self, negotiator_id: str, state):
        p = 17
        step = state.step

        #change translate_state_act, self.state_act[0:2]
        global Q_propose_matrix
        global Q_respond_matrix
        step = state.step
        if negotiator_id in self.name_list:
            opponent_number = self.name_list.index(negotiator_id)
            state_act_letter = self.state_act[opponent_number]
            new_state = [int(step), state_act_letter[1], 0, 0, 1, self.awi.level]
            self.update_Q_matrix(state_act_letter, new_state)
            self.state_act[opponent_number] = new_state
        else:
            self.name_list.append(negotiator_id)
            self.state_act.append([0,self.first_needs - self.secured,0,0,1, self.awi.level])
            opponent_number = self.name_list.index(negotiator_id)

        random_value = np.random.uniform(0, 1)
        global epsilon0
        if random_value < epsilon0:        #whether action is random or non random
            my_needs = self.first_needs - self.secured #needs
            ami = self.get_nmi(negotiator_id) #for unit_price
            unit_price_issue = ami.issues[UNIT_PRICE]
            value_state = min(step, 20) + 21 * (max(my_needs,-2)+2) #for unit_price and quantity
            value_act = self.find_max_column(self.Q_propose_matrix, value_state)
            if value_act == 20:
                self.state_act[opponent_number][4] = 0
                return None
            act_UnitPrice_Quantity = self.inverse_translate_act_for_propose(value_act)
            if act_UnitPrice_Quantity[0] == 0:
                self.state_act[opponent_number][3] = 0
                unit_price = unit_price_issue.min_value
            else:
                self.state_act[opponent_number][3] = 1
                unit_price = unit_price_issue.max_value
            quantity = act_UnitPrice_Quantity[1]
            self.state_act[opponent_number][4] = quantity
            offer = [-1] * 3
            offer[TIME] = self.awi.current_step
            offer[UNIT_PRICE] = unit_price
            offer[QUANTITY] = quantity
            return tuple(offer)
        else:
            #make action and change self.state_act[3:4]
            selected_number0 = random.choice(range(11))
            if selected_number0 == 0:
                self.state_act[opponent_number][4] = 0
                return None
            else:
                selected_number1 = random.choice(range(2))
                ami = self.get_nmi(negotiator_id)
                offer = [-1] * 3
                offer[QUANTITY] = selected_number0
                offer[TIME] = self.awi.current_step
                unit_price_issue = ami.issues[UNIT_PRICE]
                if selected_number1 == 0:
                    offer[UNIT_PRICE] = unit_price_issue.min_value
                else:
                    offer[UNIT_PRICE] = unit_price_issue.max_value
                self.state_act[opponent_number][3] = selected_number1
                self.state_act[opponent_number][4] = selected_number0
                return tuple(offer)

    def respond(self, negotiator_id, state, offer, source=""):
        ami = self.get_nmi(negotiator_id)
        step = state.step
        unit_price_issue = ami.issues[UNIT_PRICE]
        global Q_propose_matrix
        global Q_respond_matrix
        if negotiator_id in self.name_list:
            opponent_number = self.name_list.index(negotiator_id)
            state_act_letter = self.state_act[opponent_number]
            new_state = [int(step), state_act_letter[1], 1, offer[UNIT_PRICE], offer[QUANTITY], self.awi.level]
            if offer[UNIT_PRICE] == unit_price_issue.min_value: #for new_state[3]
                new_state[3] = 0
            else:
                new_state[3] = 1
            self.update_Q_matrix(state_act_letter, new_state)
            self.state_act[opponent_number] = new_state
        else:
            self.name_list.append(negotiator_id)
            self.state_act.append([0,self.first_needs - self.secured,1, offer[UNIT_PRICE], offer[QUANTITY], self.awi.level])
            opponent_number = self.name_list.index(negotiator_id)
            if offer[UNIT_PRICE] == unit_price_issue.min_value: #for self.state_act[3]
                self.state_act[opponent_number][3] = 0
            else:
                self.state_act[opponent_number][3] = 1

        random_value = np.random.uniform(0, 1)
        global epsilon0
        if random_value < epsilon0:        #random or non random
            step = state.step #step
            my_needs = self.first_needs - self.secured #needs

            ami = self.get_nmi(negotiator_id) #for unit_price
            unit_price_issue = ami.issues[UNIT_PRICE]
            if offer[UNIT_PRICE] == unit_price_issue.min_value:
                value_unit_price = 0
            else:
                value_unit_price = 1
            quantity = offer[QUANTITY] #quantity
            value_state = min(step, 20) + 21 * (max(my_needs,-2)+2) + 273 * value_unit_price + 546 * (quantity - 1)
            value_act = self.find_max_column(self.Q_respond_matrix, value_state)
            if value_act == 0:
                self.state_act[opponent_number][2] = 1
                return ResponseType.ACCEPT_OFFER
            elif value_act == 1:
                self.state_act[opponent_number][2] = 2
                return ResponseType.REJECT_OFFER
            else:
                self.state_act[opponent_number][2] = 3
                return ResponseType.END_NEGOTIATION
        else:
            #change self.state_act[2]
            selected_number2 = random.choice(range(3))
            if selected_number2 == 0:
                self.state_act[opponent_number][2] = 1
                return ResponseType.ACCEPT_OFFER
            elif selected_number2 < 2:
                self.state_act[opponent_number][2] = 2
                return ResponseType.REJECT_OFFER
            else:
                self.state_act[opponent_number][2] = 3
                return ResponseType.END_NEGOTIATION


    def update_Q_matrix(self, state_act, new_state):
        global Q_propose_matrix
        global Q_respond_matrix
        global learning_rate
        global discount_rate
        if state_act[2] == 0:
            row = translate_state(state_act)
            col = translate_act(state_act)
            q0 = self.Q_propose_matrix[row, col]
        else:
            row = translate_state(state_act)
            col = translate_act(state_act)
            q0 = self.Q_respond_matrix[row, col]
        reward = abs(state_act[1]) - abs(new_state[1])
        if new_state[4] != 0:
            if new_state[2] == 0:
                q1 = np.amax(self.Q_propose_matrix[translate_state(new_state)])
            else:
                q1 = np.amax(self.Q_respond_matrix[translate_state(new_state)])
        else:
            q1 = 0
        result = q0 + learning_rate * (reward + discount_rate * q1 - q0)
        if state_act[2] == 0:
            self.Q_propose_matrix[translate_state(state_act), translate_act(state_act)] = result
        else:
            self.Q_respond_matrix[translate_state(state_act), translate_act(state_act)] = result


    def inverse_translate_act_for_propose(self, value):
        if value == 20:
            return [1,0]
        elif value % 2 == 0:
            return [0,value/2]
        else:
            return [1,(value-1)/2]


    def find_max_column(self, data, row_index):
        row = data[row_index]
        max_value = np.max(row)
        max_column = np.argmax(row)
        return max_column

    def call_Q_propose_matrix(self):
        global Q_propose_matrix
        return Q_propose_matrix

    def call_Q_respond_matrix(self):
        global Q_respond_matrix
        return Q_respond_matrix

    def save_Q_propose_matrix(self):
        global Q_propose_matrix
        Q_propose_matrix = self.Q_propose_matrix

    def save_Q_respond_matrix(self):
        global Q_respond_matrix
        Q_respond_matrix = self.Q_respond_matrix
