import csv
from AgentQuantTrackerClass import AgentQuantTracker
from collections import defaultdict

with open(r'FILE_PATH_HERE', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

#print(data[0][6]) # simulation day completed (0-19)
#print(type(data[0][7])) # quantity agreed in contract
#print(data[1][11]) # day concluded at (-1 if exogenous)

TotalAgentData = []
AgentsNEQData = defaultdict(int)

for i in range(1,len(data)):
    if data[i][11] != '-1':
        for j in range(2):
            AgentsNEQData[data[i][j+2] + ',' + data[i][6]] += int(data[i][7])

for i in range(len(data)):
    if data[i][2] == 'SELLER':
        TotalAgentData += [AgentQuantTracker(data[i][3], data[i][5].split('.')[-1],
                                           0, data[i][7], AgentsNEQData[data[i][3] + ',' + data[i][6]],data[i][6])]
    elif data[i][3] == 'BUYER':
        TotalAgentData += [AgentQuantTracker(data[i][2], data[i][4].split('.')[-1],
                                           1, data[i][7], AgentsNEQData[data[i][2] + ',' + data[i][6]],data[i][6])]

QOAinstances = 0
QOAnotmissedinstances = 0
QOAabsdiff = 0
BQOAinstances = 0
BQOAabsdiff = 0
BQOAnotmissedinstances = 0
        
for agent in TotalAgentData:
    if agent.a_type == 'QuantityOrientedAgent':
        QOAinstances += 1
        QOAabsdiff += abs(int(agent.exo) - int(agent.quant))
        if abs(int(agent.exo) - int(agent.quant)) == 0:
            QOAnotmissedinstances += 1
    else:
        BQOAinstances += 1
        BQOAabsdiff += abs(int(agent.exo) - int(agent.quant))
        if abs(int(agent.exo) - int(agent.quant)) == 0:
            BQOAnotmissedinstances += 1
            
print(QOAinstances)
print(QOAnotmissedinstances)
print(QOAabsdiff)
print(BQOAinstances)
print(BQOAnotmissedinstances)
print(BQOAabsdiff)

QOAMR = 1.0 - (float(QOAnotmissedinstances) / float(QOAinstances))
BQOAMR = 1.0 - (float(BQOAnotmissedinstances) / float(BQOAinstances))

print('QOA miss rate: %f' % QOAMR)
print('QOA average miss quantity: %f' % (float(QOAabsdiff) / float(QOAinstances - QOAnotmissedinstances)))     

print('BQOA miss rate: %f' % BQOAMR)
print('BQOA average miss quantity: %f' % (float(BQOAabsdiff) / float(BQOAinstances - BQOAnotmissedinstances)))     
