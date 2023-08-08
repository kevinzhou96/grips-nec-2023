This NiceQOA agent was an attempt at exploring how more desperate behaviors in the default QuantityOrientedAgent (QOA) strategy would affect performance.

The sole change from the defualt QOA strategy to this "nicer" version is that NiceQOA **always** offers whatever price is best for its opponent. 
This change means we no longer need the distinct "best_offer" and "quick_offer" functions present in the original QOA agent. Thus, NiceQOA is by far the simplest
agent strategy among these various agents based on QOA in this repository.

In agent environments that I've run tournaments in, NiceQOA consistently performs below average and is not a very good agent. Specifically, it tends to perform
rather poorly when at level 1.
