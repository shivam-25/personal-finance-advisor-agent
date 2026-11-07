import pytest

from advisor.agents.planner import (
    INTENT_SPECS,
    Intent,
    Plan,
    PlanError,
    PlanStep,
)


def step(step_id, agent="spending_analysis", task="analyze_spending", depends=()):
    return PlanStep(step_id, "spend.summary", agent, task, depends_on=depends)


def test_topological_order_respects_dependencies():
    plan = Plan("g", Intent.SPEND_REVIEW, [
        step("s3", depends=("s2",)), step("s1"), step("s2", depends=("s1",)),
    ])
    assert [s.step_id for s in plan.order()] == ["s1", "s2", "s3"]


def test_independent_steps_are_ordered_deterministically():
    plan = Plan("g", Intent.SPEND_REVIEW, [step("sb"), step("sa")])
    assert [s.step_id for s in plan.order()] == ["sa", "sb"]


def test_cycle_is_rejected():
    plan = Plan("g", Intent.SPEND_REVIEW, [
        step("s1", depends=("s2",)), step("s2", depends=("s1",)),
    ])
    with pytest.raises(PlanError, match="cyclic"):
        plan.order()


def test_unknown_agent_is_rejected():
    plan = Plan("g", Intent.SPEND_REVIEW, [step("s1", agent="ghost")])
    with pytest.raises(PlanError, match="unknown agent"):
        plan.validate({"spending_analysis"}, max_steps=8)


def test_missing_dependency_is_rejected():
    plan = Plan("g", Intent.SPEND_REVIEW, [step("s1", depends=("nope",))])
    with pytest.raises(PlanError, match="depends on missing step"):
        plan.validate({"spending_analysis"}, max_steps=8)


def test_duplicate_step_ids_are_rejected():
    plan = Plan("g", Intent.SPEND_REVIEW, [step("s1"), step("s1")])
    with pytest.raises(PlanError, match="duplicate step ids"):
        plan.validate({"spending_analysis"}, max_steps=8)


def test_step_budget_is_enforced():
    plan = Plan("g", Intent.SPEND_REVIEW, [step(f"s{i}") for i in range(5)])
    with pytest.raises(PlanError, match="budget is 3"):
        plan.validate({"spending_analysis"}, max_steps=3)


def test_every_intent_declares_a_playbook():
    for intent, spec in INTENT_SPECS.items():
        assert spec.intent is intent
        assert spec.goal and spec.keywords
        if intent is not Intent.PREFERENCE_UPDATE:
            assert spec.required_capabilities
            assert spec.success_criteria


@pytest.mark.parametrize("message,expected", [
    ("How can I improve my monthly savings?", Intent.SAVINGS_IMPROVEMENT),
    ("Where did my money go last month?", Intent.SPEND_REVIEW),
    ("Am I over budget anywhere?", Intent.BUDGET_ALERTS),
    ("Has anything unusual happened on my account?", Intent.ANOMALY_CHECK),
    ("What subscriptions am I paying for?", Intent.SUBSCRIPTION_AUDIT),
    ("Am I on track for my goals?", Intent.GOAL_PROGRESS),
    ("Don't suggest cutting my travel spending", Intent.PREFERENCE_UPDATE),
    ("What is the weather in Mumbai?", Intent.OUT_OF_SCOPE),
])
def test_intent_routing(session, message, expected):
    turn = session.ask(message)
    assert turn.intent["intent"] == expected.value


def test_period_is_parsed_from_the_question(session):
    assert session.ask("Where did my money go last month?").intent["slots"]["month"] == "2026-10"
    assert session.ask("How much have I spent this month?").intent["slots"]["month"] == "2026-11"
    assert session.ask("Show me 2026-08 spending").intent["slots"]["month"] == "2026-08"


def test_follow_up_inherits_the_previous_period(session):
    session.ask("How much have I spent this month?")
    follow_up = session.ask("Anything unusual?")
    assert follow_up.intent["slots"]["month"] == "2026-11"
    assert follow_up.intent["slots"]["period_label"] == "the period we were discussing"


def test_a_complete_question_does_not_inherit_a_period(session):
    session.ask("How much have I spent this month?")
    fresh = session.ask("How can I improve my monthly savings?")
    assert fresh.intent["slots"]["month"] == "2026-10"


def test_category_is_extracted_from_an_alias(session):
    turn = session.ask("Break down my food spending")
    assert turn.intent["slots"]["category"] == "Food & Dining"


def test_savings_framing_wins_over_a_bare_category_word(session):
    turn = session.ask("How do I cut my food spending?")
    assert turn.intent["intent"] == Intent.SAVINGS_IMPROVEMENT.value


def test_plan_chains_analysis_before_advice(session):
    turn = session.ask("How can I improve my monthly savings?")
    steps = turn.plan["steps"]
    assert [s["agent"] for s in steps] == ["spending_analysis", "recommendation"]
    assert steps[1]["depends_on"] == [steps[0]["step_id"]]
    assert all(s["status"] == "done" for s in steps)


def test_plan_steps_carry_a_rationale(session):
    for plan_step in session.ask("How can I improve my monthly savings?").plan["steps"]:
        assert plan_step["rationale"]
        assert plan_step["capability"]


def test_reflection_triggers_a_replan_when_evidence_is_missing(session):
    turn = session.ask("Am I over budget anywhere?")
    origins = {s["origin"] for s in turn.plan["steps"]}
    assert "replan" in origins
    stages = [e["stage"] for e in turn.trace]
    assert "replan" in stages
    assert stages.count("reflect") >= 2
    assert all(s["status"] == "done" for s in turn.plan["steps"])


def test_replan_reruns_the_stalled_step(session):
    turn = session.ask("Am I over budget anywhere?")
    labels = [e["label"] for e in turn.trace]
    assert any("Re-running" in label for label in labels)
    assert turn.recommendations


def test_replan_does_not_duplicate_recommendations(session):
    turn = session.ask("Am I over budget anywhere?")
    titles = [r["title"] for r in turn.recommendations]
    assert len(titles) == len(set(titles))


def test_trace_covers_every_stage(session):
    stages = {e["stage"] for e in session.ask("How can I improve my monthly savings?").trace}
    assert {"intake", "understand", "recall", "plan", "execute",
            "reflect", "synthesize", "remember"} <= stages


def test_out_of_scope_builds_no_plan(session):
    turn = session.ask("What is the weather in Mumbai?")
    assert turn.plan["steps"] == []
    assert not turn.findings and not turn.recommendations
    assert "spending" in turn.answer.lower()


def test_preference_is_stored_and_confirmed(session):
    turn = session.ask("Don't suggest cutting my travel spending")
    assert turn.memory["profile"]["constraints"]["protected_categories"] == ["Travel"]
    assert "Travel" in turn.answer


def test_findings_are_ranked_by_severity(session):
    findings = session.ask("Am I over budget anywhere?").findings
    ranks = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    values = [ranks[f["severity"]] for f in findings]
    assert values == sorted(values, reverse=True)
