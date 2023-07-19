from QuantProportionLoops import TotalAgentData
import numpy as np

exoin = np.zeros(20)
exoout = np.zeros(20)
exoinminusout = np.zeros(20)

for agent in TotalAgentData:
    if agent.level == 0:
        exoin[int(agent.day)] += float(agent.exo)
    else:
        exoout[int(agent.day)] += float(agent.exo)
        
for j in range(len(exoin)):
    exoinminusout[j] = exoin[j] - exoout[j]
    
print(exoinminusout)

# at least from studying this one case, it looks like the system always has
# at least as much exogenous input as output. Thus, level 0 agents should be
# more desparate to unload supply. Also, when analyzing agent quantity
# performance rates, these excess supplies should be taken into account.
###########
# Second run was done and again, we have at least as much exo input as exo
# output. In fact, this time there was always strictly more.