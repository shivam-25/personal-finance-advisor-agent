"""Supabase-backed long-term memory, used when the filesystem is ephemeral.

Selected automatically when SUPABASE_URL and a key are set.
Table definition lives in docs/supabase_schema.sql.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

TABLE = os.getenv("SUPABASE_MEMORY_TABLE", "advisor_memory")


def supabase_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and _key())


def _key() -> str:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""


class SupabaseBackend:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 8.0) -> None:
        self.url = (url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.key = key or _key()
        self.timeout = timeout
        if not self.url or not self.key:
            raise RuntimeError("Supabase is not configured")

    @property
    def label(self) -> str:
        return f"supabase:{TABLE}"

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        request = urllib.request.Request(
            f"{self.url}/rest/v1{path}",
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload) if payload.strip() else None

    def load(self, customer_id: str) -> list[dict[str, Any]]:
        try:
            rows = self._request(
                "GET", f"/{TABLE}?customer_id=eq.{customer_id}&select=notes"
            )
        except Exception:
            return []
        if not rows:
            return []
        notes = rows[0].get("notes") or []
        return notes if isinstance(notes, list) else []

    def save(self, customer_id: str, notes: list[dict[str, Any]]) -> None:
        try:
            self._request(
                "POST",
                f"/{TABLE}?on_conflict=customer_id",
                [{"customer_id": customer_id, "notes": notes}],
            )
        except Exception:
            pass  # best-effort: a write failure must not break the turn
