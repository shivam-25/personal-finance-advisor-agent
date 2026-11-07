import pytest


def test_registry_exposes_every_tool(registry):
    names = registry.names()
    assert len(names) == 14
    for expected in ("fetch_transactions", "detect_spending_anomalies",
                     "simulate_savings_plan", "recall_memory"):
        assert expected in names


def test_schemas_are_function_calling_shaped(registry):
    for schema in registry.schemas():
        assert set(schema) == {"name", "description", "parameters"}
        assert schema["parameters"]["type"] == "object"
        assert isinstance(schema["parameters"]["properties"], dict)
        assert schema["description"]


def test_unknown_tool_is_rejected(registry):
    result = registry.call("not_a_tool")
    assert not result.ok
    assert "unknown tool" in result.error


def test_missing_required_argument_is_rejected(registry):
    result = registry.call("simulate_savings_plan")
    assert not result.ok
    assert "missing required argument 'reductions'" == result.error


def test_unexpected_argument_is_rejected(registry):
    result = registry.call("get_budget_status", {"nope": 1})
    assert not result.ok
    assert "unexpected argument" in result.error


def test_enum_is_enforced(registry):
    result = registry.call("remember_preference",
                           {"kind": "invalid", "key": "x", "value": "y"})
    assert not result.ok
    assert "must be one of" in result.error


def test_wrong_type_is_rejected(registry):
    result = registry.call("get_spend_trend", {"months": "six"})
    assert not result.ok
    assert "must be integer" in result.error


def test_tool_failure_is_captured_not_raised(registry):
    result = registry.call("summarize_spend_by_category", {"month": "1999-01"})
    assert not result.ok
    assert "no transactions found" in result.error
    with pytest.raises(Exception):
        result.unwrap()


def test_profile_tool_reports_the_calendar(registry):
    data = registry.call("get_customer_profile").unwrap()
    assert data["name"] == "Ananya Sharma"
    assert data["current_month"] == "2026-11"
    assert data["last_complete_month"] == "2026-10"


def test_fetch_transactions_filters_by_category(registry):
    data = registry.call("fetch_transactions", {"category": "Food & Dining"}).unwrap()
    assert data["count"] > 0
    assert {t["category"] for t in data["transactions"]} == {"Food & Dining"}


def test_summary_defaults_to_last_complete_month(registry):
    data = registry.call("summarize_spend_by_category").unwrap()
    assert data["month"] == "2026-10"
    assert data["total_spend"] > 0


def test_anomaly_tool_reports_all_three_signals(registry):
    data = registry.call("detect_spending_anomalies", {"month": "2026-10"}).unwrap()
    assert data["amount_outliers"] and data["duplicate_charges"] and data["category_spikes"]
    assert data["anomaly_count"] == (
        len(data["amount_outliers"]) + len(data["duplicate_charges"])
        + len(data["category_spikes"])
    )


def test_simulation_ignores_unknown_categories(registry):
    data = registry.call("simulate_savings_plan",
                         {"reductions": {"Food & Dining": 10, "Yachting": 50}}).unwrap()
    assert data["ignored_categories"] == ["Yachting"]
    assert data["monthly_saving"] > 0


def test_subscription_filter_drops_essentials(registry):
    everything = registry.call("list_recurring_subscriptions").unwrap()
    discretionary = registry.call("list_recurring_subscriptions",
                                  {"discretionary_only": True}).unwrap()
    assert discretionary["count"] < everything["count"]
    assert "Rent" not in {s["category"] for s in discretionary["subscriptions"]}


def test_memory_tools_round_trip(registry):
    registry.call("remember_preference",
                  {"kind": "preference", "key": "Travel", "value": "protected"}).unwrap()
    recalled = registry.call("recall_memory", {"query": "travel"}).unwrap()
    assert recalled["constraints"]["protected_categories"] == ["Travel"]


def test_every_call_is_traced(registry):
    registry.call("get_customer_profile")
    registry.call("not_a_tool")
    trace = registry.trace()
    assert len(trace) == 2
    assert trace[0]["ok"] and not trace[1]["ok"]
    assert all("latency_ms" in row and "called_by" in row for row in trace)
