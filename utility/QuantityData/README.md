This QuantityData repository contains 3 files related to analyzing various points about quantity in the SCML world.

The "AgentQuantTrackerClass" file contains a simple class that records an agents
name, type, level, exogenous quantity, total quantity sum of negotiated contracts, and simulation day that these quantities were recorded.

The file "QuantProportionLoops" calculates (for a single world simulation) the percent of simulation days the agent mismatched exogenous 
quantity and negotiated quantities, as well as the average absolute value of mismatched instances. The code in this file can be easily changed
to calculate things like average quantity mismatch per day rather than average quantity mismatch on days in which a mismatch occured.
**Note:** the code in this file was based on running simulations with only 2 agent types. For calculating the above-mentioned quantity
measures for more than 2 agents, the code file needs to be adjusted (likely with loops, but it should be easy to change).

The file "ExogenousQuantAnal" contains a simple code for determining values related solely to exogenous contracts
from running the QuantProportionLoops code. The version here gives the difference between total exogenous input
and exogenous output. If using this information in agent development, the configuration of the corresponding world
should be taken into account.
