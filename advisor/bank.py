"""Mock core-banking API: the single boundary the agents read customer data through."""

from __future__ import annotations

import json
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from .domain import Transaction

DATASET = Path(__file__).parent / "data" / "bank_dataset.json"


class BankAPIError(RuntimeError):
    pass


class CustomerNotFound(BankAPIError):
    pass


@lru_cache(maxsize=1)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class BankAPI:
    """Read-only facade over the mock dataset."""

    def __init__(self, dataset: Path | str = DATASET, latency_ms: int = 0) -> None:
        self._path = str(dataset)
        self._latency_ms = latency_ms

    def _raw(self) -> dict[str, Any]:
        if self._latency_ms:
            time.sleep(self._latency_ms / 1000)
        return _load(self._path)

    @property
    def as_of(self) -> date:
        return date.fromisoformat(self._raw()["as_of"])

    def list_customers(self) -> list[dict[str, Any]]:
        return [
            {k: p[k] for k in ("customer_id", "name", "segment", "city")}
            for p in self._raw()["profiles"]
        ]

    def get_profile(self, customer_id: str) -> dict[str, Any]:
        for profile in self._raw()["profiles"]:
            if profile["customer_id"] == customer_id:
                return dict(profile)
        raise CustomerNotFound(f"no such customer: {customer_id}")

    def get_transactions(
        self,
        customer_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Transaction]:
        rows = self._raw()["transactions"].get(customer_id)
        if rows is None:
            raise CustomerNotFound(f"no such customer: {customer_id}")
        out = []
        for row in rows:
            posted = date.fromisoformat(row["posted_on"])
            if start and posted < start:
                continue
            if end and posted > end:
                continue
            out.append(Transaction.from_dict(row))
        return out

    def get_budgets(self, customer_id: str) -> dict[str, float]:
        budgets = self._raw()["budgets"].get(customer_id)
        if budgets is None:
            raise CustomerNotFound(f"no such customer: {customer_id}")
        return dict(budgets)

    def get_goals(self, customer_id: str) -> list[dict[str, Any]]:
        goals = self._raw()["goals"].get(customer_id)
        if goals is None:
            raise CustomerNotFound(f"no such customer: {customer_id}")
        return [dict(g) for g in goals]
