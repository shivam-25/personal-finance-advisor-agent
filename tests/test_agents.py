from advisor.agents.alerting import AlertingAgent
from advisor.agents.base import AgentRequest
from advisor.agents.recommendation import RecommendationAgent
from advisor.agents.spending import SpendingAnalysisAgent
from advisor.domain import Severity


def analyse(registry, memory, llm, month="2026-10"):
    agent = SpendingAnalysisAgent(registry, memory, llm)
    return agent.handle(AgentRequest("analyze_spending", {"month": month}))


def test_spending_agent_produces_evidence(registry, memory, llm):
    response = analyse(registry, memory, llm)
    assert response.status == "ok"
    assert response.data["summary"]["total_spend"] > 0
    assert response.findings
    assert "summarize_spend_by_category" in response.tool_calls
    assert response.reasoning


def test_spending_agent_separates_one_offs_from_habits(registry, memory, llm):
    response = analyse(registry, memory, llm)
    assert "Shopping" in response.data["one_offs"]
    assert response.data["one_offs"]["Shopping"]["merchant"] == "Croma"
    behavioural = {m["category"] for m in response.data["behavioural_increases"]}
    assert "Food & Dining" in behavioural
    assert "Shopping" not in behavioural


def test_one_off_increase_is_not_flagged_as_high_severity(registry, memory, llm):
    response = analyse(registry, memory, llm)
    shopping = next(f for f in response.findings if f.title.startswith("Shopping up"))
    assert shopping.severity is Severity.LOW
    assert "one_off" in shopping.tags


def test_spending_agent_publishes_to_working_memory(registry, memory, llm):
    analyse(registry, memory, llm)
    assert memory.working.has("spend.analysis")


def test_spending_agent_fails_cleanly_on_empty_period(registry, memory, llm):
    response = analyse(registry, memory, llm, month="1999-01")
    assert response.status == "failed"
    assert not response.ok


def test_category_deep_dive_requires_a_category(registry, memory, llm):
    agent = SpendingAnalysisAgent(registry, memory, llm)
    response = agent.handle(AgentRequest("analyze_category", {}))
    assert response.status == "failed"
    assert response.unmet_needs == ["category"]


def test_category_deep_dive_returns_top_merchants(registry, memory, llm):
    agent = SpendingAnalysisAgent(registry, memory, llm)
    response = agent.handle(AgentRequest("analyze_category", {"category": "Food & Dining"}))
    assert response.status == "ok"
    assert response.data["top_merchants"]
    assert response.data["series"]


def test_alerting_agent_raises_ranked_alerts(registry, memory, llm):
    response = AlertingAgent(registry, memory, llm).handle(AgentRequest("check_alerts"))
    assert response.status == "ok"
    assert response.alerts
    ranks = [a.severity.rank for a in response.alerts]
    assert ranks == sorted(ranks, reverse=True)


def test_alerting_agent_projects_a_partial_month(registry, memory, llm):
    response = AlertingAgent(registry, memory, llm).handle(AgentRequest("check_alerts"))
    projected = [a for a in response.alerts if a.triggered_by == "projection"]
    assert projected
    assert "Projection confidence" in projected[0].message


def test_alerting_agent_suppresses_acknowledged_alerts(registry, memory, llm):
    agent = AlertingAgent(registry, memory, llm)
    before = agent.handle(AgentRequest("check_alerts"))
    target = before.alerts[0].category
    memory.profile.remember("acknowledged_alert", target, "ack")
    after = AlertingAgent(registry, memory, llm).handle(AgentRequest("check_alerts"))
    assert target not in {a.category for a in after.alerts}
    assert len(after.alerts) < len(before.alerts)


def test_alerting_agent_reuses_upstream_anomalies(registry, memory, llm):
    analysis = analyse(registry, memory, llm)
    response = AlertingAgent(registry, memory, llm).handle(
        AgentRequest("check_alerts", evidence={"spending_analysis": analysis.data})
    )
    assert "detect_spending_anomalies" not in response.tool_calls
    assert any("Reused anomaly evidence" in line for line in response.reasoning)


def test_recommendation_agent_declares_unmet_need_without_evidence(registry, memory, llm):
    response = RecommendationAgent(registry, memory, llm).handle(
        AgentRequest("recommend_savings")
    )
    assert response.status == "partial"
    assert response.unmet_needs == ["spend.summary"]
    assert not response.recommendations


def test_recommendation_agent_quantifies_every_action(registry, memory, llm):
    analysis = analyse(registry, memory, llm)
    response = RecommendationAgent(registry, memory, llm).handle(
        AgentRequest("recommend_savings", evidence={"spending_analysis": analysis.data})
    )
    assert response.recommendations
    assert "simulate_savings_plan" in response.tool_calls
    for rec in response.recommendations:
        assert rec.monthly_impact >= 0
        assert rec.action and rec.rationale
    impacts = [r.monthly_impact for r in response.recommendations]
    assert impacts == sorted(impacts, reverse=True)


def test_recommendation_agent_skips_one_off_categories(registry, memory, llm):
    analysis = analyse(registry, memory, llm)
    response = RecommendationAgent(registry, memory, llm).handle(
        AgentRequest("recommend_savings", evidence={"spending_analysis": analysis.data})
    )
    assert "Shopping" not in {r.category for r in response.recommendations}
    assert any("one-off" in line for line in response.reasoning)


def test_recommendation_agent_respects_protected_categories(registry, memory, llm):
    analysis = analyse(registry, memory, llm)
    memory.profile.remember("preference", "Food & Dining", "protected")
    response = RecommendationAgent(registry, memory, llm).handle(
        AgentRequest("recommend_savings", evidence={"spending_analysis": analysis.data})
    )
    assert "Food & Dining" not in {r.category for r in response.recommendations}
    assert any("protected" in line for line in response.reasoning)


def test_recommendation_agent_honours_declined_advice(registry, memory, llm):
    analysis = analyse(registry, memory, llm)
    memory.profile.remember("declined_recommendation", "subscriptions", "declined")
    response = RecommendationAgent(registry, memory, llm).handle(
        AgentRequest("recommend_savings", evidence={"spending_analysis": analysis.data})
    )
    assert "Subscriptions" not in {r.category for r in response.recommendations}


def test_goal_review_reports_shortfalls(registry, memory, llm):
    response = RecommendationAgent(registry, memory, llm).handle(
        AgentRequest("review_goals", {"month": "2026-10"})
    )
    assert response.status == "ok"
    assert len(response.data["goals"]) == 2
    assert response.findings
    assert memory.working.has("goals")


def test_agent_cards_declare_capabilities(registry, memory, llm):
    for cls in (SpendingAnalysisAgent, AlertingAgent, RecommendationAgent):
        card = cls(registry, memory, llm).card()
        assert card["capabilities"] and card["tasks"] and card["description"]
