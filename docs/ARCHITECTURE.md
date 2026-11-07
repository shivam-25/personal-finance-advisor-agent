# Architecture

## Layers

```
  ┌──────────────────────────────────────────────────────────────┐
  │ INTERFACE      web/index.html (SPA)   advisor/cli.py         │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ TRANSPORT      advisor/server.py — FastAPI, 8 endpoints      │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ RUNTIME        advisor/orchestrator.py — sessions, wiring    │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ ORCHESTRATION  advisor/agents/planner.py                     │
  │                intents · plans · reflection · synthesis      │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ SPECIALISTS    spending.py   alerting.py   recommendation.py │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ TOOLS          advisor/tools.py — 14 schema-validated tools  │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ DOMAIN         analytics.py · categorizer.py · domain.py     │
  └───────────────────────────┬──────────────────────────────────┘
  ┌───────────────────────────▼──────────────────────────────────┐
  │ DATA           bank.py → advisor/data/bank_dataset.json      │
  └──────────────────────────────────────────────────────────────┘

  CROSS-CUTTING   memory.py (3 tiers)  trace.py  llm.py  config.py
```

Each layer only depends downward. An agent cannot reach the dataset without
going through a tool, and no agent knows it is being served over HTTP.

## Design decisions

### Capability routing, not agent names

An intent declares the capabilities it needs; agents advertise capabilities
they provide. The planner matches the two.

```python
Intent.SAVINGS_IMPROVEMENT -> required_capabilities = ("spend.summary", "advice.savings")
SpendingAnalysisAgent.capabilities = ("spend.summary", "spend.trend", ...)
RecommendationAgent.capabilities  = ("advice.savings", "advice.budget", ...)
```

Adding a fifth agent, or moving a capability between agents, requires no change
to planner logic. Playbooks live in the `INTENT_SPECS` table, so a new intent is
a data entry, not a new branch.

### Plans are validated before they run

`Plan.validate()` rejects unknown agents, missing or duplicate step ids,
dependency cycles (Kahn sort) and plans over the step budget. A malformed plan
fails loudly at planning time instead of halfway through execution.

### Agents ask for what they are missing

Rather than letting an agent guess, `AgentResponse.unmet_needs` lets it tell
the planner what evidence it lacks. The Recommendation Agent dispatched without
spending evidence returns `partial` with `unmet_needs = ["spend.summary"]`. The
planner's reflect stage sees this, appends a step, runs it, and re-runs the
stalled agent. This is what makes the orchestration adaptive rather than a
fixed pipeline, and it is bounded by `ADVISOR_MAX_REPLANS`.

### The LLM cannot produce a number

The deterministic core always runs. The model is given a fact sheet of
already-computed figures and asked only to phrase them, under instructions not
to re-derive anything. Intent classification is similar: rules decide, and the
model is consulted only when the rule engine's margin is genuinely thin. With
no API key the prototype loses only prose polish, which is why the whole test
suite runs offline.

### Separating habit from one-off

The single most important piece of financial judgement in the prototype.
October Shopping is up 592%, but 78.9% of it is one ₹42,990 purchase at Croma.
A naive system recommends cutting Shopping by 20%. This one reclassifies the
category as a one-off, demotes the finding to low severity, excludes it from
advice, and says why in its reasoning. The Food & Dining rise of 78.6% — spread
across many Swiggy and Zomato orders — is kept as behavioural and becomes the
headline recommendation.

### Projections state their own confidence

A straight-line projection on day 7 of a month is noise. `budget_status`
blends the current month's daily rate with the trailing three-month rate,
weighted by elapsed days, and labels the result `low`, `medium` or `high`. The
Alerting Agent demotes severity when confidence is low, so the customer is not
alarmed by arithmetic.

### Memory as a storage seam

`ProfileMemory` talks to a `MemoryBackend`. `JsonFileBackend` keeps the
prototype dependency-free; `SupabaseBackend` is selected automatically when
`SUPABASE_URL` is present, because a serverless host gives each request a fresh
filesystem. Writes are best-effort — a storage failure degrades memory, it does
not break the turn.

## Data flow for one turn

```
message
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ UNDERSTAND   rule scorer → intent + confidence          │
│              confidence < 0.55 → LLM arbitration        │
│              slot filling, conversation fallback        │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ RECALL       recall_memory tool → constraints           │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ PLAN         capabilities → steps → deps → validate     │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ EXECUTE      topological order                          │
│              parents' data → AgentRequest.evidence      │
│              agents call tools; failure → one retry     │
│              results published to working memory        │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ REFLECT      success criteria met? unmet needs?   ──no──┐
└───────────────┬─────────────────────────────────────────┘│
                │ yes                            ┌─────────▼────────┐
                │                                │ REPLAN           │
                │                                │ append steps,    │
                │                                │ rerun stalled    │
                │                                └─────────┬────────┘
                │                                          │
                │◀─────────────────────────────────────────┘
                ▼                                   (max 2 rounds)
┌─────────────────────────────────────────────────────────┐
│ SYNTHESIZE   merge · de-duplicate · rank                │
│              template answer, optional LLM phrasing     │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│ REMEMBER     write slots and offered advice to profile  │
└───────────────┬─────────────────────────────────────────┘
                ▼
        answer + plan + trace + findings + alerts + recommendations
```

## Trace as a first-class output

Every stage appends to a `Trace`. The turn returned to the caller carries the
intent and its scores, the validated plan with each step's rationale and
status, every tool call with arguments and latency, each agent's own reasoning
lines, and a memory snapshot. The UI renders it and the CLI prints it, so the
reasoning is inspectable rather than asserted.

## Extending it

| Goal | Change needed |
|---|---|
| New intent | add an `IntentSpec` to `INTENT_SPECS` |
| New agent | implement `Agent`, declare capabilities, add to `SPECIALISTS` |
| New tool | one `@registry.register` block in `tools.py` |
| Real bank data | reimplement `BankAPI`; nothing above it changes |
| Different model | set `ADVISOR_MODEL` |

## Production gaps

This is a prototype. Reaching production would need authentication and
per-customer authorisation on every tool call, encryption of the memory store,
PII redaction before any text reaches a model provider, audit retention of
traces, idempotency on advice delivery, and human review of regulated advice.
