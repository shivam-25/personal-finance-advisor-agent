# Agent workflow

Three real turns, taken from the running prototype.

## Agents and their responsibilities

| Agent | Owns | Capabilities | Will not do |
|---|---|---|---|
| **Planner** | the turn: intent, plan, execution, reflection, synthesis | `orchestration` | compute a financial figure itself |
| **Spending Analysis** | the transaction book: enrichment, aggregation, trends, anomalies | `spend.summary` `spend.trend` `spend.comparison` `spend.anomalies` | give advice |
| **Alerting** | what must be said now: breaches, projections, anomalies | `alert.budget` `alert.projection` `alert.anomaly` | present a weak projection as fact |
| **Recommendation** | quantified actions | `advice.savings` `advice.budget` `advice.subscriptions` `advice.goals` | advise without evidence, or touch a protected category |

## Turn 1 — a plain two-step plan

**"How can I improve my monthly savings?"**

```
[    1ms] INTAKE      Query received
[    2ms] UNDERSTAND  Intent: savings_improvement (confidence 0.90, rules)
                      savings_improvement: keywords ['improve', 'savings']
[    2ms] UNDERSTAND  Slots resolved
                      month=2026-10; period_label=last month
[    2ms] RECALL      Long-term memory: 0 relevant note(s)
[    2ms] PLAN        Plan built: 2 step(s)
                      1. s1 -> spending_analysis.analyze_spending
                         why: spend.summary is required for
                              'savings_improvement'; spending_analysis
                              advertises it
                      2. s2 -> recommendation.recommend_savings  (after s1)
                         why: advice.savings is required ...; recommendation
                              advertises it and needs s1 first
[    7ms] EXECUTE     s1: spending_analysis.analyze_spending -> ok
                      2026-10: ₹1,26,891 spent. +71.8% against 2026-09.
                      9 anomaly signal(s) detected.
                      - Enriched 60 transactions; 100.0% categorised
                      tools: categorize_transactions,
                             summarize_spend_by_category, compare_periods,
                             detect_spending_anomalies, get_spend_trend
[   11ms] EXECUTE     s2: recommendation.recommend_savings -> ok
                      4 action(s) worth ₹7,095 a month (₹85,140 a year).
                      - Skipped Shopping: the increase is a one-off ₹42,990
                        purchase at Croma, not a habit.
                      - Simulated the plan: ₹4,330/month
                      tools: compute_savings_capacity, simulate_savings_plan,
                             list_recurring_subscriptions, get_financial_goals
[   11ms] REFLECT     Success criteria satisfied
                      criteria: spend.analysis, recommendations | missing: none
[   11ms] SYNTHESIZE  Merged 13 finding(s), 0 alert(s), 4 recommendation(s)
[   11ms] REMEMBER    Wrote 2 note(s) to long-term memory
```

The dependency matters: `recommend_savings` declares it needs `spend.summary`,
so the planner orders the analyst first and passes its output as evidence. Note
the analyst found Shopping up 592%, and the recommender deliberately refused to
act on it.

## Turn 2 — reflection forces a re-plan

**"Am I over budget anywhere?"**

The `budget_alerts` playbook asks for `alert.budget` and `advice.budget`. It
does not ask for spending analysis — so the planner starts without it, and has
to discover mid-turn that it is needed.

```
[    2ms] PLAN        Plan built: 2 step(s)
                      1. s1 -> alerting.check_alerts
                      2. s2 -> recommendation.recommend_budget_fix
[    5ms] EXECUTE     s1: alerting.check_alerts -> ok
                      7 active alert(s) for 2026-11.
[    5ms] EXECUTE     s2: recommendation.recommend_budget_fix -> partial
                      Waiting on spending analysis before recommending.
                      - No spending evidence on the blackboard - requesting
                        spend.summary from the planner before advising.
[    5ms] REFLECT     Evidence incomplete
                      criteria: alerts, recommendations | missing:
                      recommendations | agent requests: spend.summary
[    5ms] REPLAN      Plan extended with 1 step(s)
                      + r3 -> spending_analysis.analyze_spending
                        (Re-plan: 'spend.summary' was requested by a
                         specialist or needed for an unmet success criterion.)
[   10ms] EXECUTE     r3: spending_analysis.analyze_spending -> ok
[   10ms] REPLAN      Re-running s2 now that its evidence exists
                      recommendation.recommend_budget_fix previously needed
                      spend.summary
[   15ms] EXECUTE     s2: recommendation.recommend_budget_fix -> ok
[   15ms] REFLECT     Success criteria satisfied
[   15ms] SYNTHESIZE  Merged 20 finding(s), 7 alert(s), 4 recommendation(s)
```

The agent did not fail and it did not guess. It reported precisely what it was
missing, and the planner repaired its own plan.

## Turn 3 — memory changes the advice

**"Don't suggest cutting my travel spending"**

```
[    1ms] UNDERSTAND  Intent: preference_update (confidence 0.94, rules)
[    1ms] UNDERSTAND  Slots resolved   category=Travel
[    2ms] REMEMBER    Stored 1 preference(s)
                      preference:Travel=protected
```

No plan is built; this turn only writes memory. Asking the savings question
again afterwards produces a different plan outcome:

```
[    2ms] RECALL      Long-term memory: 1 relevant note(s)
                      protected: Travel | declined: none
[   11ms] EXECUTE     s2: recommendation.recommend_savings -> ok
                      - Memory constraints applied - protected: ['Travel']
                      - Skipped Travel: customer marked it protected.
```

Travel disappears from the plan, the remaining actions are re-ranked, and the
reason is stated rather than hidden.

## Intent playbooks

| Intent | Required capabilities | Success criteria |
|---|---|---|
| `savings_improvement` | `spend.summary`, `advice.savings` | `spend.analysis`, `recommendations` |
| `spend_review` | `spend.summary` | `spend.analysis` |
| `budget_alerts` | `alert.budget`, `advice.budget` | `alerts`, `recommendations` |
| `anomaly_check` | `spend.anomalies` | `spend.anomalies` |
| `category_deep_dive` | `spend.trend` | `spend.category` |
| `subscription_audit` | `advice.subscriptions` | `recommendations` |
| `goal_progress` | `advice.goals` | `goals` |
| `financial_health` | `spend.summary`, `alert.budget`, `advice.savings` | all three |
| `preference_update` | — | — |

## Failure handling

| Situation | Behaviour |
|---|---|
| Tool call fails | recorded on the trace, agent degrades to `partial` and continues |
| Agent fails | retried once with its period constraint relaxed |
| Dependency produced nothing | dependent step is skipped, not fed garbage |
| Evidence still missing after re-plans | answers with what it has and says the trace shows the gap |
| No capability advertised for an intent | capability skipped and noted on the trace |
| Plan malformed | rejected by the validator before execution |
| LLM unavailable or slow | falls back to the deterministic writer |
| Memory store unwritable | memory degrades to in-process; the turn completes |
| Question out of scope | declines and lists what it can do, builds no plan |
