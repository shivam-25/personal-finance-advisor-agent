"""Value objects shared by the agents and tools."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

CATEGORIES = (
    "Food & Dining",
    "Groceries",
    "Shopping",
    "Travel",
    "Transport",
    "Bills & Utilities",
    "Entertainment",
    "Health & Wellness",
    "Rent",
    "Education",
    "Investments",
    "Income",
    "Transfers",
    "Fees & Charges",
    "Uncategorised",
)

DISCRETIONARY = frozenset(
    {"Food & Dining", "Shopping", "Entertainment", "Travel", "Transport"}
)
ESSENTIAL = frozenset(
    {"Rent", "Bills & Utilities", "Groceries", "Health & Wellness", "Education"}
)
NON_SPEND = frozenset({"Income", "Investments", "Transfers"})


@dataclass(slots=True)
class Transaction:
    txn_id: str
    account_id: str
    posted_on: date
    description: str
    merchant: str
    amount: float
    direction: Direction
    channel: str
    category: str = "Uncategorised"
    category_confidence: float = 0.0
    is_recurring: bool = False

    @property
    def signed_amount(self) -> float:
        return -self.amount if self.direction is Direction.DEBIT else self.amount

    @property
    def month(self) -> str:
        return self.posted_on.strftime("%Y-%m")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Transaction":
        return cls(
            txn_id=raw["txn_id"],
            account_id=raw["account_id"],
            posted_on=date.fromisoformat(raw["posted_on"]),
            description=raw["description"],
            merchant=raw["merchant"],
            amount=float(raw["amount"]),
            direction=Direction(raw["direction"]),
            channel=raw["channel"],
            category=raw.get("category", "Uncategorised"),
            category_confidence=float(raw.get("category_confidence", 0.0)),
            is_recurring=bool(raw.get("is_recurring", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["posted_on"] = self.posted_on.isoformat()
        out["direction"] = self.direction.value
        return out


@dataclass(slots=True)
class Finding:
    """A single piece of evidence produced by a specialist agent."""

    finding_id: str
    source: str
    kind: str
    title: str
    detail: str
    severity: Severity = Severity.INFO
    evidence: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["severity"] = self.severity.value
        out["tags"] = list(self.tags)
        return out


@dataclass(slots=True)
class Recommendation:
    rec_id: str
    title: str
    rationale: str
    action: str
    category: str
    monthly_impact: float
    effort: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Alert:
    alert_id: str
    title: str
    message: str
    severity: Severity
    category: str
    triggered_by: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["severity"] = self.severity.value
        return out


def money(amount: float) -> str:
    """Indian-format currency string, e.g. 1234567.5 -> '₹12,34,568'."""
    whole = int(round(abs(amount)))
    text = str(whole)
    if len(text) > 3:
        head, tail = text[:-3], text[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        text = ",".join(groups + [tail])
    return f"{'-' if amount < 0 else ''}₹{text}"


def pct(value: float) -> str:
    return f"{value:+.1f}%"


def utc_now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()
