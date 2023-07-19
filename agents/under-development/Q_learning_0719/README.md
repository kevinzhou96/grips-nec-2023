You need two things "pip install negmas==0.9.8", "pip install scml==0.5.6".

The problem of this Agent are lack of learning, reward is not utility function but only absolute of quantity and there are some erros in learning

I mayby put explanation of my code by monday.\
This is a little explanation of letter version code\
https://docs.google.com/document/d/1GRpdaOJP5q461WiuUdl1h-pyE534zRIDYtiFakeuYVA/edit?usp=sharing

Not all code is putted, so prease refer google colab\
https://colab.research.google.com/drive/1h68RoxJLNEvi5G7WS8FX86ObQEIdQOkC?usp=sharing\

\
"""\
abstruct of my code in google colab\
If you change some code in my google colab, The change is not preserved. So, you can change code feel free.

Other Settings: make output short in google colab. if you don't use google colab, you don't use "Other Settings"

install: install negmas and scml

import: import some module for caluculating and negmas

make environment: I copied and pasted from official cite.

AgentClass: define some Agent\
  Agent for Q-learning: can learn in execution\
  Agent by Q-value: take action depend on Q_propose_matrix and Q_respond_matrix\
  original Agent: be made by other people\
  experimental Agent: be changed for experiment

prepare initial state of Q-value: prepare initial state of Q_propose_matrix and Q_respond_matrix\
  initialize Q-Value: define Q_propose_matrix and Q_respond_matrix\
  edit Q-Value: make Agent do best action when best action is obvious (such as No needs)

function: define functions for treating Q_propose_matrix and Q_respond_matrix

try anac2023OneShot: try anac2023OneShot\
  prepare: install and define function for trying anac2023OneShot\
  execution: define some parameters and execute\
"""
