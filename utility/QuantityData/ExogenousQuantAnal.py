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
