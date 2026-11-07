"""Ordered record of the planner's decisions for one turn, rendered by the UI and CLI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .domain import utc_now


@dataclass
class TraceEvent:
    stage: str
    label: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=utc_now)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label": self.label,
            "detail": self.detail,
            "data": self.data,
            "at": self.at,
            "elapsed_ms": self.elapsed_ms,
        }


class Trace:
    STAGES = (
        "intake",
        "understand",
        "recall",
        "plan",
        "execute",
        "reflect",
        "replan",
        "synthesize",
        "remember",
    )

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._started = time.perf_counter()

    def add(
        self, stage: str, label: str, detail: str = "", **data: Any
    ) -> TraceEvent:
        event = TraceEvent(
            stage=stage,
            label=label,
            detail=detail,
            data=data,
            elapsed_ms=int((time.perf_counter() - self._started) * 1000),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def render(self) -> str:
        """Human-readable form for the CLI."""
        lines = []
        for event in self._events:
            lines.append(f"[{event.elapsed_ms:>5}ms] {event.stage.upper():<11} {event.label}")
            if event.detail:
                for chunk in event.detail.split("\n"):
                    lines.append(f"{'':>9}   {chunk}")
        return "\n".join(lines)
