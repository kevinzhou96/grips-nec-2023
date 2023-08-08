Acceptance Strategy\
  if step >= 18 or level 0:\
    the best price combination of offers which the most fulfill needs\
  else:\
    the most offers fulfilling needs which have the best price\
    if new_needs > 0 and current_permit_needs > 0 and remain regotiating_partners:\
      the best price combination of offers which the most fulfill current_permit_needs\
      reduce current_permit_needs by the quantity acquired at this time

  """\
  permit_needs: \
    at first of simulation, permit_needs = 0\
    renew permit_needs as below every 10 days based on the average of remaining final needs for the last 10 day\
  current_permit_needs:\
    at first of each days, current_permit_needs = permit_needs\
  """

Reject and End Strategy\
  after deciding apceptance, \
  if new needs > 0 or remaining negotiating partners:\
    reject and proposals\
  else:\
    end negotiation

Proposal Strategy\
  unit_price = best unit price\
  if number of negotiating partners = 1:\
    quantity = needs\
  elif needs <= 9:\
    quantity = math.ceil(needs/2)\
  else:\
    quantity = math.ceil(needs/2) + 1
