import json

import pytest

from advisor.cli import main
from advisor.config import settings
from advisor.orchestrator import AdvisorRuntime


def test_llm_is_disabled_for_tests():
    assert settings.llm_enabled is False


def test_answers_are_reproducible_without_an_llm(backend):
    question = "How can I improve my monthly savings?"
    first = AdvisorRuntime(backend=backend).ask("CUST001", question)
    second = AdvisorRuntime(backend=backend).ask("CUST001", question)
    assert first.answer == second.answer
    assert [r["monthly_impact"] for r in first.recommendations] == [
        r["monthly_impact"] for r in second.recommendations
    ]


def test_no_figure_is_invented_when_the_llm_is_off(session):
    turn = session.ask("How can I improve my monthly savings?")
    simulated = {
        line["category"]: line["monthly_saving"]
        for line in turn.agent_responses["s2"]["data"]["simulation"]["lines"]
    }
    for rec in turn.recommendations:
        if rec["category"] in simulated:
            assert rec["monthly_impact"] == pytest.approx(simulated[rec["category"]])


def test_full_conversation_keeps_context(session):
    session.ask("Where did my money go last month?")
    session.ask("What about this month?")
    turn = session.ask("Anything unusual?")
    assert turn.intent["slots"]["month"] == "2026-11"
    assert len(turn.memory["conversation"]) == 6


@pytest.mark.parametrize("customer", ["CUST001", "CUST002", "CUST003"])
def test_every_customer_can_be_advised(runtime, customer):
    turn = runtime.ask(customer, "How can I improve my monthly savings?")
    assert turn.answer
    assert turn.findings
    assert all(s["status"] == "done" for s in turn.plan["steps"])


def test_disciplined_customer_gets_fewer_alerts(runtime):
    overspender = runtime.ask("CUST002", "Am I over budget anywhere?")
    saver = runtime.ask("CUST003", "Am I over budget anywhere?")
    assert len(saver.alerts) < len(overspender.alerts)


def test_cli_ask_emits_json(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("ADVISOR_MEMORY_DIR", str(tmp_path))
    assert main(["ask", "Where did my money go last month?", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["intent"]["intent"] == "spend_review"
    assert payload["trace"]


def test_cli_lists_agents(capsys):
    assert main(["agents"]) == 0
    out = capsys.readouterr().out
    for name in ("planner", "spending_analysis", "alerting", "recommendation"):
        assert name in out


def test_cli_lists_customers(capsys):
    assert main(["customers"]) == 0
    assert "Ananya Sharma" in capsys.readouterr().out


def test_cli_ask_prints_a_trace(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("ADVISOR_MEMORY_DIR", str(tmp_path))
    assert main(["ask", "Am I over budget anywhere?"]) == 0
    out = capsys.readouterr().out
    assert "REPLAN" in out and "EXECUTE" in out
