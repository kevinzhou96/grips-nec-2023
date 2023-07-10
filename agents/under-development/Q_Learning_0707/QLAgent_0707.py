"""You need two files Q_propose_matrix.csv, Q_respond_matrix.csv. """

class OneShotAgent_By_Q_Learning_0707(OneShotAgent):
    """Based on OneShotAgent"""

    def init(self):
        self.Q_propose_matrix = np.loadtxt("Q_propose_matrix.csv", delimiter=",")
        self.Q_respond_matrix = np.loadtxt("Q_respond_matrix.csv", delimiter=",")

    def before_step(self):  # Resets counts
        self.secured = 0
        self.first_needs = self.awi.current_exogenous_input_quantity + self.awi.current_exogenous_output_quantity

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state):
        step = state.step #step
        my_needs = self.first_needs - self.secured #needs

        ami = self.get_nmi(negotiator_id) #for unit_price
        unit_price_issue = ami.issues[UNIT_PRICE]
        value_state = min(step, 20) + 21 * (max(my_needs,-2)+2) #for unit_price and quantity
        value_act = self.find_max_column(self.Q_propose_matrix, value_state)
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
        return tuple(offer)

    def respond(self, negotiator_id, state, offer, source=""):
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
            return ResponseType.ACCEPT_OFFER
        elif value_act == 1:
            return ResponseType.REJECT_OFFER
        else:
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
        max_column = np.argmax(row)
        return max_column
