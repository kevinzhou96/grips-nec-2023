This is an updated form of the GreedyQOA agent in this same completed agents directory.

The primary change between GreedyQOA and this "HalfGreedy" version is that in our counteroffers to opponents, we offer half of the last quantity they offered rather than all of it.
This is done as an attempt to account for the likely decrease in their desired quantities from accepting other negotiators' offers between our own negotiation rounds (between the two of us).

In the spaces I've checked, this agent performs notably better than the GreedyQOA agent, however it's still consistently losing to agents like the defualt QuantityOrientedAgent strategy.
