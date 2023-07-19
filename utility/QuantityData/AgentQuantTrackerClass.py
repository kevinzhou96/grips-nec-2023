class AgentQuantTracker:
    def __init__(self, name, a_type, level, exo, quant, day): 
        self.name = name
        self.a_type = a_type
        self.level = level
        self.exo = exo
        self.quant = quant
        self.day = day

    def __str__(self):
        return "Name: %s, Agent Type: %s, Production level: %d Exogenous Quant: %s, Non-Exogenous Quant: %s Day: %s" % \
     (self.name, self.a_type, self.level, self.exo, self.quant, self.day)