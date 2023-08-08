"""ByQValueAgent.py"""
Acceptance Strategy\
  if step >= 19 or level 0:\
    the best price combination of offers which the most fulfill needs\
  else:\
    the most offers fulfilling needs which have the best price

Reject and End Strategy\
  after deciding apceptance,\
  if new needs > 0 or remaining negotiating partners:\
    reject and proposals\
  else:\
    end negotiation

Proposal Strategy\
  according to Q_value_matrix.csv


"""Q_value_matrix-8.csv"""\
Sample Q_value_matrix in the process of learning


"""AgentForQLearning.py"""\
agents updating Q-value-matrix throughout execution


"""prepare.py"""\
perform various installations and imports


"""choice_of_propose.py"""\
define choices of actions to optimize through learning


"""settings.py"""\
define the parameters to be used for learning and the initial state of the Q-value-matrix


"""execute.py"""\
execute
