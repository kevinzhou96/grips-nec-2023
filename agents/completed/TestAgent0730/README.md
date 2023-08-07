Acceptance Strategy
  if step >= 19 or level 0:
    the best price combination of offers which the most fulfill needs
  else:
    the most offers fulfilling needs which have the best price

Reject and End Strategy
  after deciding apceptance, 
  if new needs > 0 or remaining negotiating partners:
    reject and proposals
  else:
    end negotiation

Proposal Strategy
  unit_price = best unit price
  if number of negotiating partners = 1:
    quantity = needs
  elif needs <= 9:
    quantity = math.ceil(needs/2)
  else:
    quantity = math.ceil(needs/2) + 1
