# g-RIPS Sendai 2023 - NEC Team 
This is the GitHub Repo for the g-RIPS Sendai 2023 NEC Project on Automated Reasoning for Supply Chain Management.

In this project, we implement several negotiating agents for the ANAC Supply Chain Management League (SCML), as well as some utility scripts to aid in agent development and analysis. 

# Agents
Our agents are broken into two categories. 
The first category consists of agents which are based on various heuristics.
These agents are:
- GreedyQOA
- HalfGreedyQOA
- NiceQOA
- Nice/QOA hybrid
- Half-desperate/Half-greedy hybrid
- QOA/Half-greedy hybrid

The second category consists of agents which partly or wholly use reinforcement learning to determine their strategies. 
These agents are:
- RLAgent-v0 
    - a family of pure RL agents, along with training and testing scripts
- ByQValueAgent
    - an agent trained with a significantly restricted observation and action space
- HeuristicAgent
    - an agent with a fixed strategy that was primarily informed by the results of ByQValueAgent
- AdaptEnvironmentAgent
    - an agent designed to improve upon HeuristicAgent based on results of testing


# The Team

The g-RIPS Sendai 2023 NEC team consists of:
- Isaac Brown (The Ohio State University)
- Soto Hisakawa (Kyushu University)
- Soichiro Sato (Musashino University)
- Kevin Zhou (University of Illinois at Chicago)

This project was conducted under the supervision of:
- Academic mentor: Dr. Masaki Ogawa (Tohoku University)
- Industry mentor: Dr. Yasser Mohammad (NEC)

# Acknowledgments

The team would like to thank NEC, AIST, and Dr. Mohammad for sponsoring this wonderful project. Additionally, we would like to thank the Insitute of Pure and Applied Mathematics at UCLA (IPAM) and the Advanced Institute for Materials Research at Tohoku University (AIMR) for organizing the 2023 g-RIPS program. In particular, we are extremely grateful to Dr. Christian Ratsch, Deputy Director of IPAM, and Dr. Hiroshi Suito, Deputy Director of AIMR, for all the work they put into making g-RIPS Sendai 2023 happen. 
