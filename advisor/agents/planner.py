"""Planner Agent: the coordinator, and the agent built in depth.

Six stages per turn:

  UNDERSTAND   resolve the query to an intent and fill its slots, using the
               conversation for anything left implicit ("and last month?")
  RECALL       load long-term memory so constraints are known before advice
  PLAN         select agents by capability, order by dependency, validate
  EXECUTE      run in topological order, feeding each agent its parents' output
  REFLECT      test the evidence against the intent's success criteria and
               re-plan if an agent declared an unmet need
  SYNTHESIZE   merge, de-duplicate and rank, then write back to memory

Deterministic by default. The LLM only arbitrates a low-confidence intent and
phrases the final answer; it never produces a figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .. import analytics as an
from ..domain import Alert, Finding, Recommendation, Severity, money
from ..trace import Trace
from .base import Agent, AgentRequest, AgentResponse


class Intent(str, Enum):
    SPEND_REVIEW = "spend_review"
    SAVINGS_IMPROVEMENT = "savings_improvement"
    BUDGET_ALERTS = "budget_alerts"
    ANOMALY_CHECK = "anomaly_check"
    CATEGORY_DEEP_DIVE = "category_deep_dive"
    SUBSCRIPTION_AUDIT = "subscription_audit"
    GOAL_PROGRESS = "goal_progress"
    FINANCIAL_HEALTH = "financial_health"
    PREFERENCE_UPDATE = "preference_update"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class IntentSpec:
    """Declarative playbook. Adding an intent does not change planner code."""

    intent: Intent
    goal: str
    #: capabilities that must be covered, in order
    required_capabilities: tuple[str, ...]
    #: working-memory keys that must exist for the answer to be complete
    success_criteria: tuple[str, ...]
    keywords: tuple[str, ...]
    phrases: tuple[str, ...] = ()


INTENT_SPECS: dict[Intent, IntentSpec] = {
    Intent.SAVINGS_IMPROVEMENT: IntentSpec(
        Intent.SAVINGS_IMPROVEMENT,
        "Find and quantify ways to free up money each month.",
        ("spend.summary", "advice.savings"),
        ("spend.analysis", "recommendations"),
        ("save", "saving", "savings", "reduce", "cut", "lower", "improve", "optimise",
         "optimize", "spare", "surplus", "afford"),
        ("improve my savings", "save more", "spend less", "cut down"),
    ),
    Intent.SPEND_REVIEW: IntentSpec(
        Intent.SPEND_REVIEW,
        "Explain where the money went and how that compares to before.",
        ("spend.summary",),
        ("spend.analysis",),
        ("spend", "spent", "spending", "where", "breakdown", "summary", "report",
         "month", "categories", "much"),
        ("where did my money go", "how much did i spend", "spending summary"),
    ),
    Intent.BUDGET_ALERTS: IntentSpec(
        Intent.BUDGET_ALERTS,
        "Report budget breaches, anything on track to breach, and how to fix it.",
        ("alert.budget", "advice.budget"),
        ("alerts", "recommendations"),
        ("budget", "limit", "alert", "alerts", "warning", "over", "exceed",
         "breach", "threshold", "overspend", "overspending"),
        ("am i over budget", "budget status", "any alerts"),
    ),
    Intent.ANOMALY_CHECK: IntentSpec(
        Intent.ANOMALY_CHECK,
        "Surface unusual, duplicate or suspicious transactions.",
        ("spend.anomalies",),
        ("spend.anomalies",),
        ("unusual", "strange", "odd", "anomaly", "anomalies", "suspicious",
         "duplicate", "fraud", "charged", "twice", "wrong", "unexpected"),
        ("anything unusual", "charged twice", "weird transaction"),
    ),
    Intent.CATEGORY_DEEP_DIVE: IntentSpec(
        Intent.CATEGORY_DEEP_DIVE,
        "Break one category down by merchant and trend.",
        ("spend.trend",),
        ("spend.category",),
        ("food", "dining", "shopping", "travel", "transport", "groceries",
         "entertainment", "bills", "utilities", "health", "rent", "detail",
         "merchant", "merchants", "breakdown"),
        ("tell me about my food spending", "break down shopping"),
    ),
    Intent.SUBSCRIPTION_AUDIT: IntentSpec(
        Intent.SUBSCRIPTION_AUDIT,
        "List recurring subscriptions and what cancelling them would recover.",
        ("advice.subscriptions",),
        ("recommendations",),
        ("subscription", "subscriptions", "recurring", "autopay", "mandate",
         "netflix", "renewal", "membership"),
        ("what subscriptions do i have", "recurring payments"),
    ),
    Intent.GOAL_PROGRESS: IntentSpec(
        Intent.GOAL_PROGRESS,
        "Report progress toward goals and what is needed to stay on track.",
        ("advice.goals",),
        ("goals",),
        ("goal", "goals", "target", "emergency", "fund", "trip", "on track",
         "deposit", "retirement"),
        ("am i on track", "my goals", "emergency fund"),
    ),
    Intent.FINANCIAL_HEALTH: IntentSpec(
        Intent.FINANCIAL_HEALTH,
        "Give a full picture: spending, risks and what to do next.",
        ("spend.summary", "alert.budget", "advice.savings"),
        ("spend.analysis", "alerts", "recommendations"),
        ("health", "overall", "overview", "everything", "review", "advice",
         "doing", "financially", "position", "checkup", "check-up"),
        ("how am i doing", "financial health", "full review", "overall picture"),
    ),
    Intent.PREFERENCE_UPDATE: IntentSpec(
        Intent.PREFERENCE_UPDATE,
        "Record a preference or a declined recommendation, then confirm it.",
        (),
        (),
        ("dont", "don't", "do not", "never", "stop", "keep", "protect",
         "non-negotiable", "remember", "ignore", "prefer"),
        ("don't suggest", "stop telling me", "remember that i"),
    ),
}

#: below this rule-engine confidence the planner asks the LLM to arbitrate
LLM_ARBITRATION_THRESHOLD = 0.55

CATEGORY_ALIASES = {
    "food": "Food & Dining",
    "food delivery": "Food & Dining",
    "dining": "Food & Dining",
    "restaurant": "Food & Dining",
    "eating out": "Food & Dining",
    "swiggy": "Food & Dining",
    "zomato": "Food & Dining",
    "grocery": "Groceries",
    "groceries": "Groceries",
    "shopping": "Shopping",
    "travel": "Travel",
    "trips": "Travel",
    "transport": "Transport",
    "commute": "Transport",
    "fuel": "Transport",
    "cab": "Transport",
    "entertainment": "Entertainment",
    "movies": "Entertainment",
    "bills": "Bills & Utilities",
    "utilities": "Bills & Utilities",
    "health": "Health & Wellness",
    "fitness": "Health & Wellness",
    "gym": "Health & Wellness",
    "rent": "Rent",
}

#: capability -> (agent capability name, task to dispatch)
CAPABILITY_TASKS = {
    "spend.summary": "analyze_spending",
    "spend.trend": "analyze_category",
    "spend.comparison": "analyze_spending",
    "spend.anomalies": "detect_anomalies",
    "alert.budget": "check_alerts",
    "alert.projection": "check_alerts",
    "alert.anomaly": "check_alerts",
    "advice.savings": "recommend_savings",
    "advice.budget": "recommend_budget_fix",
    "advice.subscriptions": "audit_subscriptions",
    "advice.goals": "review_goals",
}

#: evidence a task wants on the blackboard before it runs
TASK_DEPENDS_ON_CAPABILITY = {
    "recommend_savings": ("spend.summary",),
    "recommend_budget_fix": ("spend.summary", "alert.budget"),
    "check_alerts": (),
    "review_goals": (),
}

#: working-memory key written by each task, used to test success criteria
TASK_PRODUCES = {
    "analyze_spending": "spend.analysis",
    "analyze_category": "spend.category",
    "detect_anomalies": "spend.anomalies",
    "check_alerts": "alerts",
    "recommend_savings": "recommendations",
    "recommend_budget_fix": "recommendations",
    "audit_subscriptions": "recommendations",
    "review_goals": "goals",
}


@dataclass
class PlanStep:
    step_id: str
    capability: str
    agent: str
    task: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    rationale: str = ""
    status: str = "pending"  # pending | done | failed | skipped
    attempts: int = 0
    origin: str = "initial"  # initial | replan

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "capability": self.capability,
            "agent": self.agent,
            "task": self.task,
            "params": self.params,
            "depends_on": list(self.depends_on),
            "rationale": self.rationale,
            "status": self.status,
            "attempts": self.attempts,
            "origin": self.origin,
        }


class PlanError(RuntimeError):
    pass


@dataclass
class Plan:
    goal: str
    intent: Intent
    steps: list[PlanStep] = field(default_factory=list)

    def validate(self, known_agents: set[str], max_steps: int) -> None:
        if len(self.steps) > max_steps:
            raise PlanError(f"plan has {len(self.steps)} steps, budget is {max_steps}")
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise PlanError("duplicate step ids in plan")
        for step in self.steps:
            if step.agent not in known_agents:
                raise PlanError(f"step {step.step_id} routes to unknown agent '{step.agent}'")
            for parent in step.depends_on:
                if parent not in ids:
                    raise PlanError(f"step {step.step_id} depends on missing step '{parent}'")
        self.order()  # raises on a cycle

    def order(self) -> list[PlanStep]:
        """Kahn topological sort; raises if the dependency graph has a cycle."""
        by_id = {s.step_id: s for s in self.steps}
        pending = {s.step_id: set(s.depends_on) for s in self.steps}
        ordered: list[PlanStep] = []
        while pending:
            ready = sorted(k for k, deps in pending.items() if not deps)
            if not ready:
                raise PlanError(f"cyclic dependency among {sorted(pending)}")
            for step_id in ready:
                ordered.append(by_id[step_id])
                del pending[step_id]
            for deps in pending.values():
                deps.difference_update(ready)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "intent": self.intent.value,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ResolvedQuery:
    intent: Intent
    confidence: float
    method: str
    slots: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 2),
            "method": self.method,
            "slots": self.slots,
            "scores": {k: round(v, 3) for k, v in sorted(
                self.scores.items(), key=lambda kv: -kv[1])[:4]},
            "notes": self.notes,
        }


class PlannerAgent(Agent):
    name = "planner"
    description = (
        "Understands the request, selects specialist agents by capability, "
        "orders and validates a plan, executes it with reflection and "
        "re-planning, and synthesises the final answer."
    )
    capabilities = ("orchestration",)
    tasks = ("handle_query",)

    def __init__(self, tools, memory, llm, agents: list[Agent], settings) -> None:
        super().__init__(tools, memory, llm)
        self.agents = agents
        self.settings = settings
        self._by_capability: dict[str, Agent] = {}
        for agent in agents:
            for capability in agent.capabilities:
                self._by_capability.setdefault(capability, agent)

    # =================================================================
    def run(self, message: str, trace: Trace) -> dict[str, Any]:
        self.memory.start_turn()
        trace.add("intake", "Query received", message,
                  customer=self.tools.context.customer_id,
                  as_of=self.tools.context.as_of.isoformat())

        resolved = self._understand(message, trace)
        self.memory.conversation.add("user", message, intent=resolved.intent.value)

        constraints = self._recall(message, resolved, trace)

        if resolved.intent is Intent.PREFERENCE_UPDATE:
            return self._handle_preference(message, resolved, trace)
        if resolved.intent is Intent.OUT_OF_SCOPE:
            return self._handle_out_of_scope(resolved, trace)

        plan = self._build_plan(resolved, trace)
        responses = self._execute(plan, resolved, trace)
        responses = self._reflect(plan, resolved, responses, trace)
        return self._synthesize(message, resolved, plan, responses, constraints, trace)

    # == stage 1: understand ==========================================
    def _understand(self, message: str, trace: Trace) -> ResolvedQuery:
        resolved = self._score_intent(message)
        trace.add(
            "understand",
            f"Intent: {resolved.intent.value} (confidence {resolved.confidence:.2f}, "
            f"{resolved.method})",
            "\n".join(resolved.notes),
            **resolved.to_dict(),
        )

        # Only pay for an LLM call when the rule engine is genuinely unsure.
        if resolved.confidence < LLM_ARBITRATION_THRESHOLD and self.llm.available:
            refined = self._llm_intent(message, resolved)
            if refined:
                trace.add(
                    "understand",
                    f"LLM disambiguation: {refined.intent.value} "
                    f"(confidence {refined.confidence:.2f})",
                    "\n".join(refined.notes),
                    **refined.to_dict(),
                )
                resolved = refined

        resolved.slots = self._fill_slots(message, resolved)
        trace.add(
            "understand",
            "Slots resolved",
            "; ".join(f"{k}={v}" for k, v in resolved.slots.items()) or "defaults used",
            slots=resolved.slots,
        )
        return resolved

    def _score_intent(self, message: str) -> ResolvedQuery:
        text = message.lower()
        tokens = set(re.findall(r"[a-z']+", text))
        scores: dict[str, float] = {}
        notes: list[str] = []

        for intent, spec in INTENT_SPECS.items():
            score = 0.0
            hits = tokens & set(spec.keywords)
            score += 1.0 * len(hits)
            matched_phrases = [p for p in spec.phrases if p in text]
            score += 2.5 * len(matched_phrases)
            if score:
                scores[intent.value] = score
                if matched_phrases:
                    notes.append(f"{intent.value}: phrase match {matched_phrases}")
                elif hits:
                    notes.append(f"{intent.value}: keywords {sorted(hits)}")

        if not scores:
            # Nothing matched. Prefer continuing the current thread over refusing.
            previous = self.memory.conversation.last_intent()
            if previous and self._is_follow_up(text):
                notes.append(f"no keywords matched; continuing previous intent '{previous}'")
                return ResolvedQuery(Intent(previous), 0.5, "conversation_carryover",
                                     notes=notes)
            notes.append("no intent keywords matched")
            return ResolvedQuery(Intent.OUT_OF_SCOPE, 0.3, "rules", notes=notes)

        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        confidence = scores[best] / total if total else 0.0
        runner_up = sorted(scores.values(), reverse=True)[1:2]
        margin = scores[best] - (runner_up[0] if runner_up else 0.0)

        # A bare category word plus no other signal is a deep dive, but the
        # same word inside a savings question is not.
        if (
            Intent(best) is Intent.CATEGORY_DEEP_DIVE
            and margin <= 1.0
            and Intent.SAVINGS_IMPROVEMENT.value in scores
        ):
            notes.append(
                "category keyword present but savings framing dominates; "
                "treating as savings_improvement"
            )
            best = Intent.SAVINGS_IMPROVEMENT.value

        # Calibrated so a clear single-intent match lands near 0.75, an exact
        # phrase match near 0.95, and a genuine two-way tie below the 0.55
        # threshold that hands the decision to the language model.
        confidence = min(0.95, 0.35 + 0.15 * margin + 0.25 * confidence)
        if confidence < LLM_ARBITRATION_THRESHOLD:
            notes.append(
                f"margin {margin:.1f} over the runner-up is too small to be sure; "
                "escalating to the language model if one is configured"
            )
        return ResolvedQuery(Intent(best), confidence, "rules", scores=scores, notes=notes)

    def _is_follow_up(self, text: str) -> bool:
        return bool(
            re.search(r"\b(and|what about|how about|also|then|that|those|it)\b", text)
        ) or len(text.split()) <= 5

    def _llm_intent(self, message: str, current: ResolvedQuery) -> ResolvedQuery | None:
        options = [i.value for i in INTENT_SPECS]
        catalogue = "\n".join(
            f"- {spec.intent.value}: {spec.goal}" for spec in INTENT_SPECS.values()
        )
        result = self.llm.complete(
            "intent_classification",
            [
                {
                    "role": "system",
                    "content": (
                        "You route retail-banking customer messages to one intent. "
                        "Use only the listed intents.\n\n" + catalogue
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent conversation:\n{self.memory.conversation.transcript()}\n\n"
                        f"Latest message: {message}"
                    ),
                },
            ],
            tool={
                "name": "route_intent",
                "description": "Pick the single best intent for the message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string", "enum": options},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["intent", "confidence", "reason"],
                },
            },
            max_tokens=250,
        )
        args = result.tool_args or {}
        if not result.ok or args.get("intent") not in options:
            return None
        return ResolvedQuery(
            Intent(args["intent"]),
            min(float(args.get("confidence", 0.7)), 0.95),
            "llm",
            scores=current.scores,
            notes=[f"LLM: {args.get('reason', '')}"],
        )

    def _fill_slots(self, message: str, resolved: ResolvedQuery) -> dict[str, Any]:
        """Resolve period and category, falling back to what was last discussed."""
        text = message.lower()
        context = self.tools.context
        slots: dict[str, Any] = {}
        current = context.current_month()
        if re.search(r"\b(this month|current month|so far|right now|today)\b", text):
            slots["month"] = current
            slots["period_label"] = "this month so far"
        elif re.search(r"\blast month\b|\bprevious month\b", text):
            slots["month"] = an.shift_month(current, -1)
            slots["period_label"] = "last month"
        elif match := re.search(r"\b(\d{4})-(\d{2})\b", text):
            slots["month"] = f"{match.group(1)}-{match.group(2)}"
            slots["period_label"] = slots["month"]
        elif match := re.search(r"\blast (\d+) months?\b", text):
            slots["months_back"] = int(match.group(1))
            slots["month"] = an.shift_month(current, -1)
            slots["period_label"] = f"the last {match.group(1)} months"
        else:
            # Only inherit the previous period when the message is elliptical
            # ("and shopping?"). A fully formed question gets the default.
            remembered = self._last_month_slot() if self._is_follow_up(text) else None
            slots["month"] = remembered or an.shift_month(current, -1)
            slots["period_label"] = (
                "the period we were discussing" if remembered else "last month"
            )
            if remembered:
                resolved.notes.append(f"period carried over from earlier turn: {remembered}")

        for alias, category in CATEGORY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                slots["category"] = category
                break
        if "category" not in slots and resolved.intent is Intent.CATEGORY_DEEP_DIVE:
            carried = self._last_category_slot()
            if carried:
                slots["category"] = carried
                resolved.notes.append(f"category carried over from earlier turn: {carried}")

        self.memory.working.put("slot.month", slots.get("month"))
        if slots.get("category"):
            self.memory.working.put("slot.category", slots["category"])
        return slots

    def _last_month_slot(self) -> str | None:
        note = next((n for n in self.memory.profile.all("session_slot")
                     if n.key == "month"), None)
        return note.value if note else None

    def _last_category_slot(self) -> str | None:
        note = next((n for n in self.memory.profile.all("session_slot")
                     if n.key == "category"), None)
        return note.value if note else None

    # == stage 2: recall ==============================================
    def _recall(self, message: str, resolved: ResolvedQuery, trace: Trace) -> dict[str, Any]:
        result = self.tools.call("recall_memory", {"query": message}, called_by=self.name)
        payload = result.data if result.ok else {"constraints": {}, "notes": []}
        constraints = payload.get("constraints", {})
        notes = payload.get("notes", [])
        detail = (
            "protected: "
            + (", ".join(constraints.get("protected_categories") or []) or "none")
            + " | declined: "
            + (", ".join(map(str, constraints.get("declined") or [])) or "none")
        )
        trace.add(
            "recall",
            f"Long-term memory: {len(notes)} relevant note(s)",
            detail,
            constraints=constraints,
            notes=notes,
        )
        return constraints

    # == stage 3: plan ================================================
    def _build_plan(self, resolved: ResolvedQuery, trace: Trace) -> Plan:
        spec = INTENT_SPECS[resolved.intent]
        plan = Plan(goal=spec.goal, intent=resolved.intent)
        produced: dict[str, str] = {}  # capability -> step_id

        for capability in spec.required_capabilities:
            step = self._step_for(capability, resolved, produced, origin="initial")
            if step is None:
                trace.add(
                    "plan",
                    f"No agent advertises '{capability}'",
                    "Capability skipped; the answer will note the gap.",
                    capability=capability,
                )
                continue
            plan.steps.append(step)
            produced[capability] = step.step_id

        try:
            plan.validate({a.name for a in self.agents}, self.settings.max_plan_steps)
        except PlanError as exc:
            trace.add("plan", "Plan rejected by validator", str(exc))
            raise

        ordered = plan.order()
        trace.add(
            "plan",
            f"Plan built: {len(plan.steps)} step(s)",
            "\n".join(
                f"{i + 1}. {s.step_id} -> {s.agent}.{s.task}"
                + (f"  (after {', '.join(s.depends_on)})" if s.depends_on else "")
                + f"\n   why: {s.rationale}"
                for i, s in enumerate(ordered)
            ),
            plan=plan.to_dict(),
            execution_order=[s.step_id for s in ordered],
        )
        return plan

    def _step_for(
        self,
        capability: str,
        resolved: ResolvedQuery,
        produced: dict[str, str],
        origin: str,
    ) -> PlanStep | None:
        agent = self._by_capability.get(capability)
        if agent is None:
            return None
        task = CAPABILITY_TASKS.get(capability)
        if task is None or not agent.can_handle(task):
            task = next((t for t in agent.tasks), None)
        if task is None:
            return None

        depends = tuple(
            produced[need]
            for need in TASK_DEPENDS_ON_CAPABILITY.get(task, ())
            if need in produced
        )
        params = self._params_for(task, resolved)
        rationale = (
            f"{capability} is required for '{resolved.intent.value}'; "
            f"{agent.name} advertises it"
            + (f" and needs {', '.join(depends)} first" if depends else "")
        )
        index = len(produced) + 1
        return PlanStep(
            step_id=f"s{index}" if origin == "initial" else f"r{index}",
            capability=capability,
            agent=agent.name,
            task=task,
            params=params,
            depends_on=depends,
            rationale=rationale,
            origin=origin,
        )

    def _params_for(self, task: str, resolved: ResolvedQuery) -> dict[str, Any]:
        slots = resolved.slots
        params: dict[str, Any] = {}
        if task in ("analyze_spending", "detect_anomalies", "recommend_savings",
                    "recommend_budget_fix", "analyze_category", "review_goals"):
            if slots.get("month"):
                params["month"] = slots["month"]
        if task == "check_alerts":
            # Alerts are about the live month, not the month under discussion.
            params["month"] = self.tools.context.current_month()
        if task == "analyze_category" and slots.get("category"):
            params["category"] = slots["category"]
        return params

    # == stage 4: execute =============================================
    def _execute(
        self, plan: Plan, resolved: ResolvedQuery, trace: Trace
    ) -> dict[str, AgentResponse]:
        responses: dict[str, AgentResponse] = {}
        for step in plan.order():
            if step.status in ("done", "skipped"):
                continue
            self._run_step(step, plan, responses, trace)
        return responses

    def _run_step(
        self,
        step: PlanStep,
        plan: Plan,
        responses: dict[str, AgentResponse],
        trace: Trace,
    ) -> None:
        agent = next((a for a in self.agents if a.name == step.agent), None)
        if agent is None:
            step.status = "failed"
            return

        blocked = [
            p for p in step.depends_on
            if responses.get(p) is None or not responses[p].ok
        ]
        if blocked:
            step.status = "skipped"
            trace.add(
                "execute",
                f"{step.step_id} skipped",
                f"dependency {', '.join(blocked)} did not produce usable evidence",
                step_id=step.step_id,
            )
            return

        evidence = {responses[p].agent: responses[p].data for p in step.depends_on}
        step.attempts += 1
        response = agent.handle(AgentRequest(step.task, dict(step.params), evidence))

        if response.status == "failed" and step.attempts == 1:
            # One recovery attempt with the period constraint dropped.
            relaxed = {k: v for k, v in step.params.items() if k != "month"}
            trace.add(
                "execute",
                f"{step.step_id} failed, retrying with relaxed parameters",
                f"{response.summary or 'agent reported failure'} | retry params: {relaxed or 'defaults'}",
                step_id=step.step_id,
            )
            step.attempts += 1
            response = agent.handle(AgentRequest(step.task, relaxed, evidence))

        step.status = "done" if response.ok else "failed"
        responses[step.step_id] = response
        trace.add(
            "execute",
            f"{step.step_id}: {step.agent}.{step.task} -> {response.status}",
            "\n".join(
                [response.summary]
                + [f"- {line}" for line in response.reasoning]
                + [f"tools: {', '.join(response.tool_calls)}" if response.tool_calls else ""]
            ).strip(),
            step_id=step.step_id,
            agent=step.agent,
            task=step.task,
            status=response.status,
            confidence=round(response.confidence, 2),
            tool_calls=response.tool_calls,
            findings=len(response.findings),
            unmet_needs=response.unmet_needs,
        )

    # == stage 5: reflect / re-plan ===================================
    def _reflect(
        self,
        plan: Plan,
        resolved: ResolvedQuery,
        responses: dict[str, AgentResponse],
        trace: Trace,
    ) -> dict[str, AgentResponse]:
        spec = INTENT_SPECS[resolved.intent]

        for round_index in range(self.settings.max_replans + 1):
            satisfied, missing = self._check_criteria(spec)
            unmet = {
                need
                for response in responses.values()
                for need in response.unmet_needs
                if need in CAPABILITY_TASKS
            }
            trace.add(
                "reflect",
                "Success criteria satisfied" if satisfied and not unmet
                else "Evidence incomplete",
                (
                    f"criteria: {', '.join(spec.success_criteria) or 'none'} | "
                    f"missing: {', '.join(missing) or 'none'} | "
                    f"agent requests: {', '.join(sorted(unmet)) or 'none'}"
                ),
                round=round_index,
                missing=missing,
                unmet_needs=sorted(unmet),
            )
            if satisfied and not unmet:
                return responses
            if round_index >= self.settings.max_replans:
                trace.add(
                    "replan",
                    "Re-plan budget exhausted",
                    f"Answering with partial evidence after {round_index} re-plan(s).",
                )
                return responses

            added = self._extend_plan(plan, resolved, responses, unmet, missing, trace)
            if not added:
                trace.add(
                    "replan",
                    "No further step would help",
                    "Nothing left to try; answering with what is available.",
                )
                return responses
            for step in added:
                self._run_step(step, plan, responses, trace)
            # Re-run the stalled steps first, then clear the needs they declared.
            self._rerun_dependents(plan, responses, unmet, trace)
            for response in responses.values():
                response.unmet_needs = [n for n in response.unmet_needs if n not in unmet]
        return responses

    def _check_criteria(self, spec: IntentSpec) -> tuple[bool, list[str]]:
        missing = []
        for key in spec.success_criteria:
            if self.memory.working.has(key):
                continue
            # spend.category is written under a per-category key
            if key == "spend.category" and any(
                k.startswith("spend.category.") for k in self.memory.working.keys()
            ):
                continue
            missing.append(key)
        return (not missing), missing

    def _extend_plan(
        self,
        plan: Plan,
        resolved: ResolvedQuery,
        responses: dict[str, AgentResponse],
        unmet: set[str],
        missing: list[str],
        trace: Trace,
    ) -> list[PlanStep]:
        produced = {
            s.capability: s.step_id for s in plan.steps if s.status == "done"
        }
        planned_outputs = {
            TASK_PRODUCES.get(s.task) for s in plan.steps if s.status != "failed"
        }
        wanted: list[str] = list(unmet)
        for key in missing:
            if key in planned_outputs:
                # An existing step already covers this; it is stalled, not absent,
                # and _rerun_dependents will drive it once its evidence lands.
                continue
            for capability, task in CAPABILITY_TASKS.items():
                if TASK_PRODUCES.get(task) == key and capability not in produced:
                    wanted.append(capability)
                    break

        added: list[PlanStep] = []
        for capability in dict.fromkeys(wanted):
            if capability in produced:
                continue
            if len(plan.steps) + len(added) >= self.settings.max_plan_steps:
                break
            step = self._step_for(capability, resolved, produced, origin="replan")
            if step is None:
                continue
            step.step_id = f"r{len(plan.steps) + len(added) + 1}"
            step.rationale = (
                f"Re-plan: '{capability}' was requested by a specialist or needed "
                f"for an unmet success criterion."
            )
            plan.steps.append(step)
            added.append(step)

        if added:
            trace.add(
                "replan",
                f"Plan extended with {len(added)} step(s)",
                "\n".join(f"+ {s.step_id} -> {s.agent}.{s.task}  ({s.rationale})"
                          for s in added),
                added=[s.to_dict() for s in added],
            )
        return added

    def _rerun_dependents(
        self,
        plan: Plan,
        responses: dict[str, AgentResponse],
        satisfied: set[str],
        trace: Trace,
    ) -> None:
        """Re-run steps that previously stalled waiting on evidence now present."""
        for step in plan.order():
            previous = responses.get(step.step_id)
            if previous is None or not previous.unmet_needs:
                continue
            if not set(previous.unmet_needs) & satisfied:
                continue
            source = next(
                (s.step_id for s in plan.steps
                 if s.capability in satisfied and s.status == "done"),
                None,
            )
            if source:
                step.depends_on = tuple(dict.fromkeys(step.depends_on + (source,)))
            trace.add(
                "replan",
                f"Re-running {step.step_id} now that its evidence exists",
                f"{step.agent}.{step.task} previously needed "
                f"{', '.join(previous.unmet_needs)}",
                step_id=step.step_id,
            )
            step.status = "pending"
            self._run_step(step, plan, responses, trace)

    # == stage 6: synthesize ==========================================
    def _synthesize(
        self,
        message: str,
        resolved: ResolvedQuery,
        plan: Plan,
        responses: dict[str, AgentResponse],
        constraints: dict[str, Any],
        trace: Trace,
    ) -> dict[str, Any]:
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []
        alerts: list[Alert] = []
        seen: set[str] = set()
        rec_keys: set[tuple[str, str]] = set()
        alert_ids: set[str] = set()
        duplicate_recs = 0

        for step in plan.order():
            response = responses.get(step.step_id)
            if response is None:
                continue
            for finding in response.findings:
                key = f"{finding.kind}:{finding.title}"
                if key in seen:
                    continue
                seen.add(key)
                findings.append(finding)
            for rec in response.recommendations:
                key = (rec.title, rec.category)
                if key in rec_keys:
                    duplicate_recs += 1
                    continue
                rec_keys.add(key)
                recommendations.append(rec)
            for alert in response.alerts:
                if alert.alert_id in alert_ids:
                    continue
                alert_ids.add(alert.alert_id)
                alerts.append(alert)

        findings.sort(key=lambda f: f.severity.rank, reverse=True)
        recommendations.sort(key=lambda r: r.monthly_impact, reverse=True)
        alerts.sort(key=lambda a: a.severity.rank, reverse=True)

        trace.add(
            "synthesize",
            f"Merged {len(findings)} finding(s), {len(alerts)} alert(s), "
            f"{len(recommendations)} recommendation(s)",
            f"de-duplicated {len(seen) - len(findings)} finding(s) and "
            f"{duplicate_recs} recommendation(s); ranked by severity then rupee impact",
            findings=len(findings),
            alerts=len(alerts),
            recommendations=len(recommendations),
        )

        answer = self._compose(message, resolved, findings, recommendations, alerts,
                               responses, constraints, trace)
        self.memory.conversation.add("assistant", answer)
        self._write_memory(resolved, recommendations, trace)

        return {
            "answer": answer,
            "intent": resolved.to_dict(),
            "plan": plan.to_dict(),
            "findings": [f.to_dict() for f in findings],
            "recommendations": [r.to_dict() for r in recommendations],
            "alerts": [a.to_dict() for a in alerts],
            "agent_responses": {k: v.to_dict() for k, v in responses.items()},
            "constraints": constraints,
        }

    def _compose(
        self,
        message: str,
        resolved: ResolvedQuery,
        findings: list[Finding],
        recommendations: list[Recommendation],
        alerts: list[Alert],
        responses: dict[str, AgentResponse],
        constraints: dict[str, Any],
        trace: Trace,
    ) -> str:
        deterministic = self._template_answer(resolved, findings, recommendations, alerts)
        if not self.llm.available:
            trace.add("synthesize", "Answer composed from templates",
                      "No LLM key configured; using the deterministic writer.")
            return deterministic

        facts = self._fact_sheet(resolved, findings, recommendations, alerts, constraints)
        result = self.llm.complete(
            "final_answer",
            [
                {
                    "role": "system",
                    "content": (
                        "You are a retail bank's personal finance advisor speaking to "
                        "the customer. Rewrite the supplied findings into a clear reply.\n"
                        "Rules: use only the numbers given, never invent or re-derive "
                        "any figure, keep rupee amounts exactly as written, be concise "
                        "(under 200 words), lead with the single most important point, "
                        "use short paragraphs or a tight list, and end with one concrete "
                        "next step. Do not mention agents, tools or internal mechanics."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer asked: {message}\n\n"
                        f"Verified findings:\n{facts}"
                    ),
                },
            ],
            max_tokens=600,
            temperature=0.35,
        )
        if result.ok and result.text:
            trace.add(
                "synthesize",
                "Answer phrased by the language model",
                f"model={result.call.model}, tokens={result.call.tokens}; "
                "figures were supplied as verified facts, not generated",
                model=result.call.model,
                tokens=result.call.tokens,
            )
            return result.text
        trace.add(
            "synthesize",
            "LLM unavailable, used the deterministic writer",
            (result.call.error if result.call else "unknown error") or "",
        )
        return deterministic

    def _fact_sheet(
        self,
        resolved: ResolvedQuery,
        findings: list[Finding],
        recommendations: list[Recommendation],
        alerts: list[Alert],
        constraints: dict[str, Any],
    ) -> str:
        lines = [f"Intent: {resolved.intent.value}",
                 f"Period: {resolved.slots.get('period_label', 'last month')}"]
        if alerts:
            lines.append("Alerts:")
            lines += [f"- [{a.severity.value}] {a.title}: {a.message}" for a in alerts[:4]]
        if findings:
            lines.append("Findings:")
            lines += [f"- [{f.severity.value}] {f.title}: {f.detail}" for f in findings[:6]]
        if recommendations:
            total = sum(r.monthly_impact for r in recommendations)
            lines.append(f"Recommendations (total {money(total)}/month):")
            lines += [
                f"- {r.title} ({money(r.monthly_impact)}/month, {r.effort} effort): "
                f"{r.action} Rationale: {r.rationale}"
                for r in recommendations[:4]
            ]
        protected = constraints.get("protected_categories") or []
        if protected:
            lines.append(
                f"Customer has protected these categories, do not suggest cutting them: "
                f"{', '.join(protected)}"
            )
        return "\n".join(lines)

    def _template_answer(
        self,
        resolved: ResolvedQuery,
        findings: list[Finding],
        recommendations: list[Recommendation],
        alerts: list[Alert],
    ) -> str:
        period = resolved.slots.get("period_label", "last month")
        blocks: list[str] = []

        headline = next((f for f in findings if f.kind == "spend_summary"), None)
        if headline:
            blocks.append(f"**{headline.title}** ({period})\n{headline.detail}")

        movers = [f for f in findings if f.kind == "category_increase"][:3]
        if movers:
            blocks.append(
                "**What moved**\n"
                + "\n".join(f"- {f.title} - {f.detail}" for f in movers)
            )

        urgent = [a for a in alerts if a.severity.rank >= Severity.MEDIUM.rank][:4]
        if urgent:
            blocks.append(
                "**Needs your attention**\n"
                + "\n".join(f"- {a.title}: {a.message}" for a in urgent)
            )

        goals = [f for f in findings if f.kind == "goal_progress"]
        if goals:
            blocks.append(
                "**Your goals**\n"
                + "\n".join(f"- {f.title} - {f.detail}" for f in goals)
            )

        if recommendations:
            total = sum(r.monthly_impact for r in recommendations)
            header = (
                f"**What to do next** (worth about {money(total)} a month, "
                f"{money(total * 12)} a year)"
                if total > 0
                else "**What to do next**"
            )
            blocks.append(
                header + "\n"
                + "\n".join(
                    f"{i + 1}. {r.title} - {r.action}\n   Why: {r.rationale}"
                    for i, r in enumerate(recommendations[:4])
                )
            )

        anomalies = [f for f in findings if "anomaly" in f.tags][:3]
        if anomalies and not urgent:
            blocks.append(
                "**Worth a look**\n"
                + "\n".join(f"- {f.title}: {f.detail}" for f in anomalies)
            )

        if not blocks:
            return (
                "I could not assemble enough evidence from your account to answer that "
                "properly. Could you narrow it down - a month, or a specific category?"
            )
        return "\n\n".join(blocks)

    # == preference / fallback paths ==================================
    def _handle_preference(
        self, message: str, resolved: ResolvedQuery, trace: Trace
    ) -> dict[str, Any]:
        category = resolved.slots.get("category")
        text = message.lower()
        stored: list[dict[str, Any]] = []

        if category and re.search(r"\b(don't|dont|do not|never|stop|keep|protect)\b", text):
            result = self.tools.call(
                "remember_preference",
                {
                    "kind": "preference",
                    "key": category,
                    "value": "protected",
                    "note": f'Customer said: "{message.strip()}"',
                },
                called_by=self.name,
            )
            if result.ok:
                stored.append(result.data["stored"])
        elif re.search(r"\bsubscription", text):
            result = self.tools.call(
                "remember_preference",
                {"kind": "declined_recommendation", "key": "subscriptions",
                 "value": "declined", "note": f'Customer said: "{message.strip()}"'},
                called_by=self.name,
            )
            if result.ok:
                stored.append(result.data["stored"])

        trace.add(
            "remember",
            f"Stored {len(stored)} preference(s)" if stored
            else "Nothing concrete to store",
            "; ".join(f"{s['kind']}:{s['key']}={s['value']}" for s in stored),
            stored=stored,
        )

        if stored:
            subject = stored[0]["key"]
            answer = (
                f"Noted - I will leave {subject} out of future suggestions. "
                "I will keep it in mind from now on, including in alerts and savings plans."
            )
        else:
            answer = (
                "I can remember preferences like that. Tell me which category to leave "
                'alone - for example, "don\'t suggest cutting my travel spending".'
            )
        self.memory.conversation.add("assistant", answer)
        return {
            "answer": answer,
            "intent": resolved.to_dict(),
            "plan": Plan("Record a customer preference.", resolved.intent).to_dict(),
            "findings": [],
            "recommendations": [],
            "alerts": [],
            "agent_responses": {},
            "constraints": self.memory.profile.constraints(),
            "memory_writes": stored,
        }

    def _handle_out_of_scope(self, resolved: ResolvedQuery, trace: Trace) -> dict[str, Any]:
        trace.add(
            "plan",
            "No plan built",
            "The request does not map to a supported financial capability.",
        )
        answer = (
            "I handle your spending, budgets, alerts, subscriptions and savings goals. "
            "Try asking something like \"where did my money go last month\", "
            "\"am I over budget\", or \"how can I improve my monthly savings\"."
        )
        self.memory.conversation.add("assistant", answer)
        return {
            "answer": answer,
            "intent": resolved.to_dict(),
            "plan": Plan("Clarify what the advisor can do.", resolved.intent).to_dict(),
            "findings": [],
            "recommendations": [],
            "alerts": [],
            "agent_responses": {},
            "constraints": {},
        }

    def _write_memory(
        self, resolved: ResolvedQuery, recommendations: list[Recommendation], trace: Trace
    ) -> None:
        writes = []
        if resolved.slots.get("month"):
            writes.append(self.memory.profile.remember(
                "session_slot", "month", resolved.slots["month"],
                "last period the customer asked about"))
        if resolved.slots.get("category"):
            writes.append(self.memory.profile.remember(
                "session_slot", "category", resolved.slots["category"],
                "last category the customer asked about"))
        if recommendations:
            writes.append(self.memory.profile.remember(
                "last_advice", "top_recommendation", recommendations[0].title,
                f"offered {money(recommendations[0].monthly_impact)}/month impact"))
        trace.add(
            "remember",
            f"Wrote {len(writes)} note(s) to long-term memory",
            "; ".join(f"{w.kind}:{w.key}={w.value}" for w in writes),
            writes=[w.to_dict() for w in writes],
        )
