"""Exercises the route handlers directly, so no HTTP client dependency is needed."""

import pytest
from fastapi import HTTPException

from advisor import server


@pytest.fixture(autouse=True)
def isolated_runtime(backend, monkeypatch):
    from advisor.orchestrator import AdvisorRuntime

    monkeypatch.setattr(server, "runtime", AdvisorRuntime(backend=backend))


def test_health_reports_the_dataset_date():
    health = server.health()
    assert health["status"] == "ok"
    assert health["as_of"] == "2026-11-07"
    assert health["customers"] == 3


def test_customers_include_balances():
    rows = server.customers()["customers"]
    assert len(rows) == 3
    assert all(r["monthly_income"] > 0 and r["balance"] > 0 for r in rows)


def test_capabilities_list_every_agent_and_tool():
    info = server.capabilities()
    assert {a["name"] for a in info["agents"]} == {
        "planner", "spending_analysis", "alerting", "recommendation"
    }
    assert len(info["tools"]) == 14
    assert len(info["intents"]) == 9


def test_dashboard_returns_the_overview():
    data = server.dashboard("CUST001")
    assert data["reference_month"] == "2026-10"
    assert data["current_month"] == "2026-11"
    assert data["total_spend"] > 0
    assert data["categories"] and data["budgets"] and data["goals"]
    assert data["subscriptions"]
    assert len(data["monthly_totals"]) == 6


def test_dashboard_rejects_an_unknown_customer():
    with pytest.raises(HTTPException) as exc:
        server.dashboard("NOPE")
    assert exc.value.status_code == 404


def test_chat_returns_a_full_turn():
    turn = server.chat(server.ChatRequest(
        customer_id="CUST001", message="How can I improve my monthly savings?",
        session_id="api-test"))
    assert turn["answer"]
    assert turn["intent"]["intent"] == "savings_improvement"
    assert len(turn["plan"]["steps"]) == 2
    assert turn["trace"] and turn["tool_calls"] and turn["recommendations"]
    assert turn["duration_ms"] >= 0


def test_chat_rejects_an_unknown_customer():
    with pytest.raises(HTTPException) as exc:
        server.chat(server.ChatRequest(customer_id="NOPE", message="hi"))
    assert exc.value.status_code == 404


def test_chat_request_validates_the_message():
    with pytest.raises(ValueError):
        server.ChatRequest(customer_id="CUST001", message="")


def test_reset_keeps_long_term_memory():
    request = server.ChatRequest(customer_id="CUST001", message="x", session_id="r1")
    server.chat(server.ChatRequest(customer_id="CUST001", session_id="r1",
                                   message="Don't suggest cutting my travel spending"))
    assert server.reset(request)["reset"] is True
    turn = server.chat(server.ChatRequest(customer_id="CUST001", session_id="r1",
                                          message="How can I improve my monthly savings?"))
    assert turn["memory"]["profile"]["constraints"]["protected_categories"] == ["Travel"]


def test_memory_clear_wipes_preferences():
    request = server.ChatRequest(customer_id="CUST001", message="x", session_id="r2")
    server.chat(server.ChatRequest(customer_id="CUST001", session_id="r2",
                                   message="Don't suggest cutting my travel spending"))
    assert server.clear_memory(request)["cleared"] is True
    turn = server.chat(server.ChatRequest(customer_id="CUST001", session_id="r2",
                                          message="How can I improve my monthly savings?"))
    assert turn["memory"]["profile"]["constraints"]["protected_categories"] == []


def test_suggestions_are_offered():
    assert len(server.suggestions()["suggestions"]) >= 5
