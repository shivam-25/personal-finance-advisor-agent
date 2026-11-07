from datetime import date

import pytest

from advisor import analytics as an
from advisor.domain import Direction, Transaction


def test_month_helpers():
    assert an.shift_month("2026-01", -1) == "2025-12"
    assert an.shift_month("2026-12", 1) == "2027-01"
    assert an.month_bounds("2026-02") == (date(2026, 2, 1), date(2026, 2, 28))
    assert an.month_bounds("2026-11")[1] == date(2026, 11, 30)


def test_spend_excludes_investments_and_income(transactions):
    rows = [t for t in transactions if t.month == "2026-10"]
    categories = an.by_category(rows)
    assert "Investments" not in categories
    assert "Income" not in categories
    assert an.total_income(rows) == 145000.0


def test_category_shares_sum_to_one_hundred(transactions):
    rows = [t for t in transactions if t.month == "2026-10"]
    shares = sum(v["share_pct"] for v in an.by_category(rows).values())
    assert shares == pytest.approx(100.0, abs=0.3)


def test_month_over_month_identifies_the_food_increase(transactions):
    result = an.compare_months(transactions, "2026-10", "2026-09")
    movers = {m["category"]: m for m in result["movers"]}
    assert movers["Food & Dining"]["direction"] == "up"
    assert movers["Food & Dining"]["change_pct"] > 50
    assert result["total_change_pct"] > 0


def test_zscore_finds_the_planted_outlier(transactions):
    outliers = an.amount_outliers(transactions)
    croma = next(o for o in outliers if o["merchant"] == "Croma")
    assert croma["amount"] == 42990.0
    assert croma["z_score"] > 2.5


def test_duplicate_charge_detection(transactions):
    duplicates = an.duplicate_charges([t for t in transactions if t.month == "2026-10"])
    assert len(duplicates) == 1
    assert duplicates[0]["merchant"] == "Zomato"
    assert len(duplicates[0]["txn_ids"]) == 2


def test_duplicate_detection_respects_the_window():
    rows = [
        Transaction("D1", "A1", date(2026, 10, 1), "PAY", "Shop", 500.0,
                    Direction.DEBIT, "UPI", category="Shopping"),
        Transaction("D2", "A1", date(2026, 10, 4), "PAY", "Shop", 500.0,
                    Direction.DEBIT, "UPI", category="Shopping"),
    ]
    assert an.duplicate_charges(rows, window_days=1) == []
    assert len(an.duplicate_charges(rows, window_days=5)) == 1


def test_category_spike_against_trailing_baseline(transactions):
    spikes = {s["category"]: s for s in an.category_spikes(transactions, "2026-10")}
    assert "Food & Dining" in spikes
    assert spikes["Food & Dining"]["change_pct"] > 20
    assert spikes["Food & Dining"]["excess"] > 0


def test_budget_status_projects_a_partial_month(transactions, bank):
    budgets = bank.get_budgets("CUST001")
    rows = an.budget_status(transactions, budgets, "2026-11", date(2026, 11, 7))
    assert all(r["days_elapsed"] == 7 and r["days_in_month"] == 30 for r in rows)
    assert all(r["projection_confidence"] == "medium" for r in rows)
    assert any(r["state"] == "on_track_to_breach" for r in rows)
    # projection must never fall below what is already spent
    assert all(r["projected_spend"] >= r["spent"] for r in rows)


def test_closed_month_is_not_projected(transactions, bank):
    rows = an.budget_status(transactions, bank.get_budgets("CUST001"), "2026-09",
                            date(2026, 11, 7))
    assert all(r["projection_confidence"] == "actual" for r in rows)
    assert all(r["projected_spend"] == r["spent"] for r in rows)


def test_savings_capacity_splits_the_month(transactions):
    capacity = an.savings_capacity(transactions, "2026-10", 145000.0)
    assert capacity["income"] == 145000.0
    assert capacity["essential_spend"] > 0
    assert capacity["discretionary_spend"] > 0
    assert capacity["invested"] == 20000.0
    total = (capacity["essential_spend"] + capacity["discretionary_spend"]
             + capacity["other_spend"] + capacity["invested"] + capacity["surplus"])
    assert total == pytest.approx(capacity["income"], abs=1.0)


def test_simulation_totals_the_cuts():
    result = an.simulate_reductions({"Food & Dining": 10000.0, "Shopping": 5000.0},
                                    {"Food & Dining": 15.0, "Shopping": 10.0})
    assert result["monthly_saving"] == pytest.approx(2000.0)
    assert result["annual_saving"] == pytest.approx(24000.0)
    assert result["lines"][0]["category"] == "Food & Dining"


def test_goal_projection_math():
    goal = {"goal_id": "G1", "name": "Trip", "target_amount": 100000.0,
            "saved_amount": 40000.0, "target_date": "2027-05-31"}
    projection = an.goal_projection(goal, 10000.0, date(2026, 11, 7))
    assert projection["gap"] == 60000.0
    assert projection["progress_pct"] == 40.0
    assert projection["months_remaining"] == 6
    assert projection["required_monthly"] == pytest.approx(10000.0)
    assert projection["on_track"] is True
    assert an.goal_projection(goal, 500.0, date(2026, 11, 7))["on_track"] is False
