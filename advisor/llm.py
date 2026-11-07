"""OpenRouter client.

Optional layer: it disambiguates intent and phrases the final answer. Figures
always come from the tools, so the advisor works with no API key.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import settings


@dataclass
class LLMCall:
    purpose: str
    model: str
    ok: bool
    latency_ms: int
    tokens: int = 0
    error: str | None = None


@dataclass
class LLMResult:
    text: str | None = None
    tool_args: dict[str, Any] | None = None
    call: LLMCall | None = None

    @property
    def ok(self) -> bool:
        return bool(self.call and self.call.ok)


class LLMClient:
    """Blocking JSON-over-HTTP with one retry; never raises to the caller."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.model
        self.calls: list[LLMCall] = []

    @property
    def available(self) -> bool:
        return settings.llm_enabled

    def complete(
        self,
        purpose: str,
        messages: list[dict[str, Any]],
        tool: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Return text, or arguments when `tool` forces a function call."""
        if not self.available:
            return LLMResult(call=LLMCall(purpose, self.model, False, 0, error="llm disabled"))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature,
            "reasoning": {"enabled": False},
        }
        if tool:
            payload["tools"] = [{"type": "function", "function": tool}]
            payload["tool_choice"] = {"type": "function", "function": {"name": tool["name"]}}

        started = time.perf_counter()
        last_error = "unknown error"
        for attempt in range(2):
            try:
                body = self._post("/chat/completions", payload)
                elapsed = int((time.perf_counter() - started) * 1000)
                message = body["choices"][0]["message"]
                tokens = int(body.get("usage", {}).get("total_tokens", 0))
                call = LLMCall(purpose, body.get("model", self.model), True, elapsed, tokens)
                self.calls.append(call)

                tool_args = None
                for raw in message.get("tool_calls") or []:
                    arguments = raw.get("function", {}).get("arguments") or "{}"
                    try:
                        tool_args = json.loads(arguments)
                    except json.JSONDecodeError:
                        tool_args = None
                    break
                return LLMResult(
                    text=(message.get("content") or "").strip() or None,
                    tool_args=tool_args,
                    call=call,
                )
            except Exception as exc:  # network, HTTP or schema: all non-fatal
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    time.sleep(0.6)

        elapsed = int((time.perf_counter() - started) * 1000)
        call = LLMCall(purpose, self.model, False, elapsed, error=last_error)
        self.calls.append(call)
        return LLMResult(call=call)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{settings.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/shivam-25/personal-finance-advisor-agent",
                "X-Title": "Personal Finance Advisor Agent",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=settings.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def usage(self) -> dict[str, Any]:
        return {
            "enabled": self.available,
            "model": self.model,
            "calls": len(self.calls),
            "successful": sum(1 for c in self.calls if c.ok),
            "tokens": sum(c.tokens for c in self.calls),
        }
