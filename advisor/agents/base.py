"""Agent contract.

Agents declare capabilities, which is what the planner routes on, reach the
bank only through the tool registry, and return findings plus structured data.
`unmet_needs` is how an agent asks the planner for more work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import Alert, Finding, Recommendation, Severity
from ..llm import LLMClient
from ..memory import MemoryStore
from ..tools import ToolRegistry


@dataclass
class AgentRequest:
    task: str
    params: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent: str
    task: str
    status: str = "ok"  # ok | partial | failed
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    unmet_needs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "task": self.task,
            "status": self.status,
            "summary": self.summary,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "unmet_needs": self.unmet_needs,
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "alerts": [a.to_dict() for a in self.alerts],
            "data": self.data,
        }


class Agent:
    """Shared plumbing."""

    name = "agent"
    description = "base agent"
    capabilities: tuple[str, ...] = ()
    #: tasks this agent accepts
    tasks: tuple[str, ...] = ()

    def __init__(self, tools: ToolRegistry, memory: MemoryStore, llm: LLMClient) -> None:
        self.tools = tools
        self.memory = memory
        self.llm = llm

    def can_handle(self, task: str) -> bool:
        return task in self.tasks

    def handle(self, request: AgentRequest) -> AgentResponse:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    def _call(self, response: AgentResponse, tool: str, args: dict[str, Any] | None = None) -> Any:
        """Call a tool and record it. Returns None on failure: degrading to
        `partial` is more useful to the planner than a dead turn."""
        result = self.tools.call(tool, args, called_by=self.name)
        response.tool_calls.append(tool)
        if not result.ok:
            response.status = "partial"
            response.reasoning.append(f"tool {tool} failed ({result.error}); continuing degraded")
            return None
        return result.data

    def _finding(
        self,
        response: AgentResponse,
        kind: str,
        title: str,
        detail: str,
        severity: Severity = Severity.INFO,
        evidence: dict[str, Any] | None = None,
        tags: tuple[str, ...] = (),
    ) -> Finding:
        finding = Finding(
            finding_id=f"{self.name[:2].upper()}-{len(response.findings) + 1:02d}",
            source=self.name,
            kind=kind,
            title=title,
            detail=detail,
            severity=severity,
            evidence=evidence or {},
            tags=tags,
        )
        response.findings.append(finding)
        return finding

    def card(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "tasks": list(self.tasks),
        }
