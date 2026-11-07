"""Wires bank API, tools, agents and planner together. Sessions are cached so
a follow-up question still sees the earlier conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents.alerting import AlertingAgent
from .agents.base import Agent
from .agents.planner import INTENT_SPECS, PlannerAgent
from .agents.recommendation import RecommendationAgent
from .agents.spending import SpendingAnalysisAgent
from .bank import BankAPI
from .config import settings
from .domain import utc_now
from .llm import LLMClient
from .memory import MemoryBackend, MemoryStore
from .tools import ToolContext, ToolRegistry
from .trace import Trace

SPECIALISTS = (SpendingAnalysisAgent, AlertingAgent, RecommendationAgent)


@dataclass
class AdvisorTurn:
    customer_id: str
    session_id: str
    question: str
    answer: str
    intent: dict[str, Any]
    plan: dict[str, Any]
    findings: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    agent_responses: dict[str, Any]
    trace: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    memory: dict[str, Any]
    llm: dict[str, Any]
    duration_ms: int
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "intent": self.intent,
            "plan": self.plan,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "alerts": self.alerts,
            "agent_responses": self.agent_responses,
            "trace": self.trace,
            "tool_calls": self.tool_calls,
            "memory": self.memory,
            "llm": self.llm,
            "duration_ms": self.duration_ms,
            "at": self.at,
        }


class Session:
    """One customer's conversation and the memory spanning its turns."""

    def __init__(
        self,
        customer_id: str,
        session_id: str,
        bank: BankAPI,
        backend: MemoryBackend | None = None,
    ) -> None:
        self.customer_id = customer_id
        self.session_id = session_id
        self.bank = bank
        self.memory = MemoryStore(customer_id, session_id, backend=backend)
        self.llm = LLMClient()
        self.turns: list[AdvisorTurn] = []

    def _build(self) -> tuple[PlannerAgent, ToolRegistry]:
        """Fresh registry per turn, so each turn's trace is self-contained."""
        registry = ToolRegistry(ToolContext(self.bank, self.customer_id, self.memory))
        specialists: list[Agent] = [
            cls(registry, self.memory, self.llm) for cls in SPECIALISTS
        ]
        planner = PlannerAgent(registry, self.memory, self.llm, specialists, settings)
        return planner, registry

    def ask(self, message: str) -> AdvisorTurn:
        planner, registry = self._build()
        trace = Trace()
        result = planner.run(message, trace)
        turn = AdvisorTurn(
            customer_id=self.customer_id,
            session_id=self.session_id,
            question=message,
            answer=result["answer"],
            intent=result["intent"],
            plan=result["plan"],
            findings=result["findings"],
            recommendations=result["recommendations"],
            alerts=result["alerts"],
            agent_responses=result["agent_responses"],
            trace=trace.to_list(),
            tool_calls=registry.trace(),
            memory=self.memory.snapshot(),
            llm=self.llm.usage(),
            duration_ms=trace.total_ms,
        )
        self.turns.append(turn)
        return turn

    def agent_cards(self) -> list[dict[str, Any]]:
        planner, _ = self._build()
        return [planner.card()] + [a.card() for a in planner.agents]


class AdvisorRuntime:
    """Process-wide entry point owning the session cache."""

    def __init__(self, bank: BankAPI | None = None, backend: MemoryBackend | None = None) -> None:
        self.bank = bank or BankAPI()
        self.backend = backend
        self._sessions: dict[tuple[str, str], Session] = {}

    def session(self, customer_id: str, session_id: str = "default") -> Session:
        key = (customer_id, session_id)
        if key not in self._sessions:
            self.bank.get_profile(customer_id)  # raises CustomerNotFound
            self._sessions[key] = Session(customer_id, session_id, self.bank, self.backend)
        return self._sessions[key]

    def ask(
        self, customer_id: str, message: str, session_id: str = "default"
    ) -> AdvisorTurn:
        return self.session(customer_id, session_id).ask(message)

    def reset(self, customer_id: str, session_id: str = "default") -> None:
        self._sessions.pop((customer_id, session_id), None)

    def customers(self) -> list[dict[str, Any]]:
        return self.bank.list_customers()

    def capabilities(self) -> dict[str, Any]:
        session = self.session("CUST001", "__introspect__")
        cards = session.agent_cards()
        registry = ToolRegistry(ToolContext(self.bank, "CUST001", session.memory))
        return {
            "agents": cards,
            "tools": registry.describe(),
            "intents": [
                {
                    "intent": spec.intent.value,
                    "goal": spec.goal,
                    "requires": list(spec.required_capabilities),
                }
                for spec in INTENT_SPECS.values()
            ],
            "llm": {"enabled": settings.llm_enabled, "model": settings.model},
            "as_of": self.bank.as_of.isoformat(),
        }
