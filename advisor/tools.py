"""Tool registry.

Every fact an agent asserts comes from a call made here. Tools declare a JSON
Schema, so the same definitions serve the planner and LLM function calling.
Calls are validated, timed and recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from . import analytics as an
from .bank import BankAPI, BankAPIError
from .categorizer import categorize_all, coverage, detect_recurring
from .domain import Transaction
from .memory import MemoryStore


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    tool: str
    ok: bool
    args: dict[str, Any]
    data: Any = None
    error: str | None = None
    latency_ms: int = 0
    called_by: str = "planner"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "args": self.args,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "called_by": self.called_by,
            "result_preview": _preview(self.data),
        }

    def unwrap(self) -> Any:
        if not self.ok:
            raise ToolError(f"{self.tool} failed: {self.error}")
        return self.data


def _preview(data: Any, limit: int = 240) -> str:
    text = repr(data)
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI/OpenRouter-compatible function definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolContext:
    """Everything a tool may touch."""

    def __init__(self, bank: BankAPI, customer_id: str, memory: MemoryStore) -> None:
        self.bank = bank
        self.customer_id = customer_id
        self.memory = memory
        self._txn_cache: list[Transaction] | None = None

    @property
    def as_of(self) -> date:
        return self.bank.as_of

    def transactions(self) -> list[Transaction]:
        """Categorised book, fetched once per turn."""
        if self._txn_cache is None:
            rows = self.bank.get_transactions(self.customer_id)
            self._txn_cache = categorize_all(rows)
        return self._txn_cache

    def current_month(self) -> str:
        return an.month_key(self.as_of)

    def last_complete_month(self) -> str:
        return an.shift_month(self.current_month(), -1)


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._tools: dict[str, ToolSpec] = {}
        self.calls: list[ToolResult] = []
        _register_all(self)

    def register(
        self, name: str, description: str, parameters: dict[str, Any]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = ToolSpec(name, description, parameters, handler)
            return handler

        return decorator

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[n].schema() for n in self.names()]

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": n, "description": self._tools[n].description} for n in self.names()
        ]

    def call(
        self, name: str, args: dict[str, Any] | None = None, called_by: str = "planner"
    ) -> ToolResult:
        args = dict(args or {})
        spec = self._tools.get(name)
        if spec is None:
            result = ToolResult(name, False, args, error=f"unknown tool '{name}'",
                                called_by=called_by)
            self.calls.append(result)
            return result

        problem = _validate(spec.parameters, args)
        if problem:
            result = ToolResult(name, False, args, error=problem, called_by=called_by)
            self.calls.append(result)
            return result

        started = time.perf_counter()
        try:
            data = spec.handler(self.context, **args)
            ok, error = True, None
        except (BankAPIError, ToolError, ValueError, KeyError) as exc:
            data, ok, error = None, False, f"{type(exc).__name__}: {exc}"
        latency = int((time.perf_counter() - started) * 1000)
        result = ToolResult(name, ok, args, data, error, latency, called_by)
        self.calls.append(result)
        return result

    def trace(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


def _validate(parameters: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Required keys, unknown keys, declared types, enums."""
    properties = parameters.get("properties", {})
    for key in parameters.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    expected = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, value in args.items():
        if key not in properties:
            return f"unexpected argument '{key}'"
        declared = properties[key].get("type")
        if declared in expected and not isinstance(value, expected[declared]):
            return f"argument '{key}' must be {declared}"
        choices = properties[key].get("enum")
        if choices and value not in choices:
            return f"argument '{key}' must be one of {choices}"
    return None


# Tool definitions: the only entry points agents have to the bank.

MONTH = {"type": "string", "description": "Month as YYYY-MM. Defaults to the last complete month."}


def _register_all(registry: ToolRegistry) -> None:
    @registry.register(
        "get_customer_profile",
        "Fetch the customer's KYC profile, segment, monthly income and account balances.",
        {"type": "object", "properties": {}, "required": []},
    )
    def get_customer_profile(ctx: ToolContext) -> dict[str, Any]:
        profile = ctx.bank.get_profile(ctx.customer_id)
        profile["as_of"] = ctx.as_of.isoformat()
        profile["current_month"] = ctx.current_month()
        profile["last_complete_month"] = ctx.last_complete_month()
        return profile

    @registry.register(
        "fetch_transactions",
        "Retrieve raw transactions for a date range, optionally filtered by category.",
        {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Inclusive ISO date."},
                "end": {"type": "string", "description": "Inclusive ISO date."},
                "category": {"type": "string", "description": "Restrict to one category."},
                "limit": {"type": "integer", "description": "Max rows returned."},
            },
            "required": [],
        },
    )
    def fetch_transactions(
        ctx: ToolContext,
        start: str | None = None,
        end: str | None = None,
        category: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        rows = ctx.transactions()
        if start:
            begin = date.fromisoformat(start)
            rows = [t for t in rows if t.posted_on >= begin]
        if end:
            finish = date.fromisoformat(end)
            rows = [t for t in rows if t.posted_on <= finish]
        if category:
            rows = [t for t in rows if t.category.lower() == category.lower()]
        return {
            "count": len(rows),
            "window": {"start": start, "end": end},
            "transactions": [t.to_dict() for t in rows[:limit]],
        }

    @registry.register(
        "categorize_transactions",
        "Run the merchant enrichment engine and report categorisation coverage.",
        {"type": "object", "properties": {"month": MONTH}, "required": []},
    )
    def categorize_transactions(ctx: ToolContext, month: str | None = None) -> dict[str, Any]:
        rows = ctx.transactions()
        if month:
            rows = [t for t in rows if t.month == month]
        counts: dict[str, int] = {}
        for txn in rows:
            counts[txn.category] = counts.get(txn.category, 0) + 1
        return {
            "month": month or "all",
            "transaction_count": len(rows),
            "coverage": coverage(rows),
            "category_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }

    @registry.register(
        "summarize_spend_by_category",
        "Total spend per category for a month, with share, transaction count and top merchant.",
        {"type": "object", "properties": {"month": MONTH}, "required": []},
    )
    def summarize_spend_by_category(ctx: ToolContext, month: str | None = None) -> dict[str, Any]:
        month = month or ctx.last_complete_month()
        rows = [t for t in ctx.transactions() if t.month == month]
        if not rows:
            raise ToolError(f"no transactions found for {month}")
        return {
            "month": month,
            "total_spend": an.total_spend(rows),
            "total_income": an.total_income(rows),
            "categories": an.by_category(rows),
        }

    @registry.register(
        "compare_periods",
        "Month-over-month comparison per category, ranked by absolute change.",
        {
            "type": "object",
            "properties": {
                "month": MONTH,
                "baseline": {"type": "string", "description": "Comparison month as YYYY-MM."},
            },
            "required": [],
        },
    )
    def compare_periods(
        ctx: ToolContext, month: str | None = None, baseline: str | None = None
    ) -> dict[str, Any]:
        month = month or ctx.last_complete_month()
        baseline = baseline or an.shift_month(month, -1)
        return an.compare_months(ctx.transactions(), month, baseline)

    @registry.register(
        "get_spend_trend",
        "Monthly spend totals across the available history, plus per-category series.",
        {
            "type": "object",
            "properties": {
                "months": {"type": "integer", "description": "How many recent months to return."}
            },
            "required": [],
        },
    )
    def get_spend_trend(ctx: ToolContext, months: int = 6) -> dict[str, Any]:
        rows = ctx.transactions()
        totals = an.by_month(rows)
        keys = list(totals)[-months:]
        series = an.category_by_month(rows)
        return {
            "months": keys,
            "totals": {k: totals[k] for k in keys},
            "by_category": {
                cat: {m: values.get(m, 0.0) for m in keys} for cat, values in series.items()
            },
        }

    @registry.register(
        "detect_spending_anomalies",
        "Find amount outliers (z-score), duplicate charges and category spikes for a month.",
        {
            "type": "object",
            "properties": {
                "month": MONTH,
                "z_threshold": {"type": "number", "description": "Outlier sensitivity, default 2.5."},
            },
            "required": [],
        },
    )
    def detect_spending_anomalies(
        ctx: ToolContext, month: str | None = None, z_threshold: float = 2.5
    ) -> dict[str, Any]:
        month = month or ctx.last_complete_month()
        rows = ctx.transactions()
        in_month = [t for t in rows if t.month == month]
        outliers = [o for o in an.amount_outliers(rows, z_threshold) if o["posted_on"][:7] == month]
        duplicates = [d for d in an.duplicate_charges(in_month)]
        spikes = an.category_spikes(rows, month)
        return {
            "month": month,
            "z_threshold": z_threshold,
            "amount_outliers": outliers,
            "duplicate_charges": duplicates,
            "category_spikes": spikes,
            "anomaly_count": len(outliers) + len(duplicates) + len(spikes),
        }

    @registry.register(
        "get_budget_status",
        "Budget utilisation per category with a month-end projection and breach state.",
        {"type": "object", "properties": {"month": MONTH}, "required": []},
    )
    def get_budget_status(ctx: ToolContext, month: str | None = None) -> dict[str, Any]:
        month = month or ctx.current_month()
        budgets = ctx.bank.get_budgets(ctx.customer_id)
        rows = an.budget_status(ctx.transactions(), budgets, month, ctx.as_of)
        return {
            "month": month,
            "as_of": ctx.as_of.isoformat(),
            "is_partial_month": month == ctx.current_month(),
            "budgets": rows,
            "breached": [r["category"] for r in rows if r["state"] == "breached"],
            "at_risk": [r["category"] for r in rows if r["state"] == "on_track_to_breach"],
        }

    @registry.register(
        "list_recurring_subscriptions",
        "Detect subscription-style recurring debits with monthly and annualised cost.",
        {
            "type": "object",
            "properties": {
                "discretionary_only": {
                    "type": "boolean",
                    "description": "Exclude rent, SIPs and utilities.",
                }
            },
            "required": [],
        },
    )
    def list_recurring_subscriptions(
        ctx: ToolContext, discretionary_only: bool = False
    ) -> dict[str, Any]:
        found = detect_recurring(ctx.transactions())
        if discretionary_only:
            skip = {"Rent", "Investments", "Bills & Utilities", "Income", "Transfers"}
            found = [f for f in found if f["category"] not in skip]
        return {
            "count": len(found),
            "monthly_total": round(sum(f["monthly_amount"] for f in found), 2),
            "annualised_total": round(sum(f["annualised"] for f in found), 2),
            "subscriptions": found,
        }

    @registry.register(
        "compute_savings_capacity",
        "Split a month into essential, discretionary and invested money and derive surplus.",
        {"type": "object", "properties": {"month": MONTH}, "required": []},
    )
    def compute_savings_capacity(ctx: ToolContext, month: str | None = None) -> dict[str, Any]:
        month = month or ctx.last_complete_month()
        profile = ctx.bank.get_profile(ctx.customer_id)
        return an.savings_capacity(ctx.transactions(), month, profile["monthly_income"])

    @registry.register(
        "simulate_savings_plan",
        "What-if: apply percentage cuts to categories and total the monthly/annual saving.",
        {
            "type": "object",
            "properties": {
                "reductions": {
                    "type": "object",
                    "description": 'Category to percent cut, e.g. {"Food & Dining": 15}.',
                },
                "month": MONTH,
            },
            "required": ["reductions"],
        },
    )
    def simulate_savings_plan(
        ctx: ToolContext, reductions: dict[str, float], month: str | None = None
    ) -> dict[str, Any]:
        month = month or ctx.last_complete_month()
        rows = [t for t in ctx.transactions() if t.month == month]
        baseline = {cat: info["total"] for cat, info in an.by_category(rows).items()}
        unknown = [c for c in reductions if c not in baseline]
        result = an.simulate_reductions(baseline, {k: v for k, v in reductions.items() if k in baseline})
        result["month"] = month
        result["ignored_categories"] = unknown
        return result

    @registry.register(
        "get_financial_goals",
        "Goal progress with the monthly contribution required to stay on track.",
        {
            "type": "object",
            "properties": {
                "monthly_contribution": {
                    "type": "number",
                    "description": "Assumed contribution; defaults to the latest surplus.",
                }
            },
            "required": [],
        },
    )
    def get_financial_goals(
        ctx: ToolContext, monthly_contribution: float | None = None
    ) -> dict[str, Any]:
        goals = ctx.bank.get_goals(ctx.customer_id)
        if monthly_contribution is None:
            profile = ctx.bank.get_profile(ctx.customer_id)
            capacity = an.savings_capacity(
                ctx.transactions(), ctx.last_complete_month(), profile["monthly_income"]
            )
            monthly_contribution = max(capacity["surplus"], 0.0)
        return {
            "assumed_monthly_contribution": round(monthly_contribution, 2),
            "goals": [an.goal_projection(g, monthly_contribution, ctx.as_of) for g in goals],
        }

    @registry.register(
        "recall_memory",
        "Search the customer's long-term memory for stated preferences and past decisions.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text query."},
                "kind": {"type": "string", "description": "Filter by note kind."},
            },
            "required": [],
        },
    )
    def recall_memory(
        ctx: ToolContext, query: str | None = None, kind: str | None = None
    ) -> dict[str, Any]:
        profile = ctx.memory.profile
        notes = profile.recall(query) if query else profile.all(kind)
        return {
            "query": query,
            "constraints": profile.constraints(),
            "notes": [n.to_dict() for n in notes],
        }

    @registry.register(
        "remember_preference",
        "Persist a durable fact: a protected category, a declined recommendation or a goal.",
        {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["preference", "declined_recommendation", "goal", "fact"],
                },
                "key": {"type": "string", "description": "Subject, e.g. a category name."},
                "value": {"type": "string", "description": 'Value, e.g. "protected".'},
                "note": {"type": "string", "description": "Why this was recorded."},
            },
            "required": ["kind", "key", "value"],
        },
    )
    def remember_preference(
        ctx: ToolContext, kind: str, key: str, value: str, note: str = ""
    ) -> dict[str, Any]:
        entry = ctx.memory.profile.remember(kind, key, value, note)
        return {"stored": entry.to_dict(), "constraints": ctx.memory.profile.constraints()}
