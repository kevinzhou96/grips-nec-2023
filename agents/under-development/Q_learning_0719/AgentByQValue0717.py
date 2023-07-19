"""You need two files Q_propose_matrix.csv, Q_respond_matrix.csv. """

class AgentByQValue0717(OneShotAgent):
    """Based on OneShotAgent"""

    def init(self):
        #print("init OK? (QL)")
        """Initializes some values needed for the negotiations"""
        if self.awi.level == 0:  # L0 agent
            self.n_partners = len(
                self.awi.all_consumers[1]
            )  # determines the number of l1 agents
        else:  # L1, Start the negotiations
            self.n_partners = len(
                self.awi.all_consumers[0]
            )  # determines the number of l0 agents
        self.Q_propose_matrix = np.loadtxt("Q_propose_matrix.csv", delimiter=",")
        self.Q_respond_matrix = np.loadtxt("Q_respond_matrix.csv", delimiter=",")

    def before_step(self):  # Resets counts
        #print("before_step OK? (QL)")
        self.secured = 0
        self.rejection = 0
        self.first_needs = self.awi.current_exogenous_input_quantity + self.awi.current_exogenous_output_quantity

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
        #print("propose OK? (QL)")
        step = state.step #step
        my_needs = self.first_needs - self.secured #needs
        level = self.awi.level

        ami = self.get_nmi(negotiator_id) #for unit_price
        if not ami:
            print("ami is not", ami)
            #return None
        unit_price_issue = ami.issues[UNIT_PRICE]
        value_state = min(step, 20) + 21 * (max(my_needs,-2)+2) + 273 * level #for unit_price and quantity
        if value_state >= 546:
            print("step, my_needs, level, value_state = ", step, my_needs, level, value_state)
        value_act = self.find_max_column(self.Q_propose_matrix, value_state)
        if True:
            if np.all(self.Q_propose_matrix[value_state] == 0):
                print("not learn ", [step, my_needs, 0, 0, 0, level])
        if value_act == 20:
            return None
        act_UnitPrice_Quantity = self.inverse_translate_act_for_propose(value_act)
        if act_UnitPrice_Quantity[0] == 0:
            unit_price = unit_price_issue.min_value
        else:
            unit_price = unit_price_issue.max_value
        quantity = act_UnitPrice_Quantity[1]
        offer = [-1] * 3
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = unit_price
        offer[QUANTITY] = quantity
        #print("state_act = ", [step, my_needs, 0, offer[UNIT_PRICE], offer[QUANTITY], self.awi.level])
        return tuple(offer)

    def respond(self, negotiator_id, state, offer, source=""):
        #print("respond OK? (QL)")
        step = state.step #step
        my_needs = self.first_needs - self.secured #needs
        level = self.awi.level

        ami = self.get_nmi(negotiator_id) #for unit_price
        unit_price_issue = ami.issues[UNIT_PRICE]
        if offer[UNIT_PRICE] == unit_price_issue.min_value:
            value_unit_price = 0
        else:
            value_unit_price = 1
        quantity = offer[QUANTITY] #quantity
        value_state = min(step, 20) + 21 * (max(my_needs,-2)+2) + 273 * value_unit_price + 546 * (quantity - 1) + 5460 * level
        value_act = self.find_max_column(self.Q_respond_matrix, value_state)
        if True:
            if np.all(self.Q_respond_matrix[value_state] == 0):
                print("not learn ", [step, my_needs, 1, value_unit_price, quantity, level])
        if value_act == 0:
            if my_needs <= 0:
                print("state_act ACCEPT= ", [step, my_needs, value_act + 1, value_unit_price, offer[QUANTITY], level])
                print("Q-value in this state=", self.Q_respond_matrix[value_state])
            return ResponseType.ACCEPT_OFFER
        elif value_act == 1:
            #print("state_act REJECT= ", [step, my_needs, value_act + 1, value_unit_price, offer[QUANTITY], level])
            return ResponseType.REJECT_OFFER
        else:
            #print("state_act END= ", [step, my_needs, value_act + 1, value_unit_price, offer[QUANTITY], level])
            return ResponseType.END_NEGOTIATION


    def inverse_translate_act_for_propose(self, value):
        if value == 20:
            return [1,0]
        elif value % 2 == 0:
            return [0,value//2]
        else:
            return [1,(value-1)//2]


    def find_max_column(self, data, row_index):
        row = data[row_index]
        #max_value = np.max(row)
        max_column = np.argmax(row)
        return max_column
