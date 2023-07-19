This Level_Scores.py file gives a code that will run a tournament (the beginning of the code file lists the tournament agents
and runs the tournament, these can and should be changed when you copy the code) and then sort through all score instances for
each agent and seperate them based on the level of the agent.

Agent occurences of the same type and level are logged together. The final output of the code is a chart indicating overall
tournament score and four dictionaries. The first two dictionaries give the number of world instances of each agent at level 0
and level 1 respectively. Since tournaments cycle agents through world positions in a manner which allows each agent to
play in the same positions as other agents, the number of level 0 instances should be the same for all agents and the number
of level 1 instances should be the same for all agents. Note: The number of level 0 instances and level 1 instances of the agents
need not be the same, and in most cases these values will be different. The difference between level 0 and level 1 instances
comes from the different configurations for the tournament worlds (if a configuration has 8 level 0 factories and 4 level 1
factories, there should be twice as many level 0 instances as level 1 instances).


The third and fourth dictionaries printed are what we care about. The third dictionary is the average performance of each agent 
when in level 0 across all world instances in the tournament. Similarly, the fourth dictionary is the average performance of each 
agent when in level 1 across all world instances in the tournament. Higher numbers in these dictionaries indicate better performance.