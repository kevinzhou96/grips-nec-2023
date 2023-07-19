tournament_types = [QuantityOrientedAgent,
                    GreedyQuantityOrientedAgent,
                    NiceQuantityOrientedAgent,
                    SmartQuantityOrientedAgent,
                    CCAgent,
                    QuantityOriented_HalfGreedyHybrid,
                    HalfDesparateHalfGreedyQOA
                    ]
# may take a long time++
results = anac2023_oneshot(
    competitors=tournament_types,
    n_configs=4, # number of different configurations to generate
    n_runs_per_world=1, # number of times to repeat every simulation (with agent assignment)
    n_steps = 20, # number of days (simulation steps) per simulation
    print_exceptions=True,
)

results = shorten_names(results)

results.scores.loc[:, ["agent_name", "agent_type", "score"]].head()

results.scores["level"] = results.scores.agent_name.str.split("@", expand=True).loc[:, 1]
results.scores = results.scores.sort_values("level")
sns.lineplot(data=results.scores[["agent_type", "level", "score"]],
             x="level", y="score", hue="agent_type")
plt.plot([0.0] * len(results.scores["level"].unique()), "b--")
plt.show()

results.score_stats

results.scores.loc[:, ["agent_name", "agent_type", "score"]].head()

print(type(results.scores[["agent_type", "level", "score"]].level))

level0score_total = defaultdict(float)
level0instances = defaultdict(float)
level0avg = defaultdict(float)

level1score_total = defaultdict(float)
level1instances = defaultdict(float)
level1avg = defaultdict(float)

for i in range(len(results.scores[["agent_type", "level", "score"]])):
    if results.scores[["agent_type", "level", "score"]].level[i] == '0':
        level0instances[results.scores[["agent_type", "level", "score"]].agent_type[i]] += 1.0
        level0score_total[results.scores[["agent_type", "level", "score"]].agent_type[i]] += results.scores[["agent_type", "level", "score"]].score[i]
    else:
        level1instances[results.scores[["agent_type", "level", "score"]].agent_type[i]] += 1.0
        level1score_total[results.scores[["agent_type", "level", "score"]].agent_type[i]] += results.scores[["agent_type", "level", "score"]].score[i]
    
print('Level 0 instances:', level0instances)
print('Level 1 instances:', level1instances)

for _ in level0instances:
    level0avg[_] = level0score_total[_] / level0instances[_]
    
for _ in level1instances:
    level1avg[_] = level1score_total[_] / level1instances[_]
    
level0avg_sort = {k: v for k, v in sorted(level0avg.items(), key=lambda item: item[1])}
level1avg_sort = {k: v for k, v in sorted(level1avg.items(), key=lambda item: item[1])}

print(level0avg_sort)
print(level1avg_sort)