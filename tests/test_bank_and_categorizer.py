import pytest

from advisor.bank import BankAPI, CustomerNotFound
from advisor.categorizer import categorize, categorize_all, coverage, detect_recurring
from advisor.domain import Direction, Transaction
from datetime import date


def test_dataset_loads_three_customers(bank):
    assert [c["customer_id"] for c in bank.list_customers()] == ["CUST001", "CUST002", "CUST003"]
    assert bank.as_of == date(2026, 11, 7)


def test_unknown_customer_raises(bank):
    with pytest.raises(CustomerNotFound):
        bank.get_profile("NOPE")
    with pytest.raises(CustomerNotFound):
        bank.get_transactions("NOPE")


def test_transaction_window_is_inclusive(bank):
    rows = bank.get_transactions("CUST001", date(2026, 10, 1), date(2026, 10, 31))
    assert rows
    assert all(r.posted_on.month == 10 for r in rows)


def test_six_months_of_history(transactions):
    assert len({t.month for t in transactions}) == 6


def test_categorisation_covers_the_book(transactions):
    stats = coverage(transactions)
    assert stats["categorised_pct"] == 100.0
    assert stats["low_confidence"] == 0


def test_merchant_rule_beats_keyword_rule():
    txn = Transaction("T1", "A1", date(2026, 10, 1), "SWIGGY PAYMENT", "Swiggy",
                      400.0, Direction.DEBIT, "UPI")
    assert categorize(txn).category == "Food & Dining"
    assert txn.category_confidence > 0.9


def test_keyword_fallback_for_unknown_merchant():
    txn = Transaction("T2", "A1", date(2026, 10, 1), "MONTHLY TUITION FEE",
                      "Bright Academy", 5000.0, Direction.DEBIT, "NETBANKING")
    assert categorize(txn).category == "Education"


def test_unknown_transaction_is_flagged_low_confidence():
    txn = Transaction("T3", "A1", date(2026, 10, 1), "MISC", "Zzz Unknown Ltd",
                      100.0, Direction.DEBIT, "CARD")
    assert categorize(txn).category == "Uncategorised"
    assert txn.category_confidence < 0.5


def test_salary_credit_is_income():
    txn = Transaction("T4", "A1", date(2026, 10, 1), "SALARY CREDIT", "Salaried Payroll",
                      145000.0, Direction.CREDIT, "NEFT")
    assert categorize(txn).category == "Income"


def test_recurring_detection_finds_subscriptions(transactions):
    found = {r["merchant"]: r for r in detect_recurring(transactions)}
    assert {"Netflix", "Spotify", "Amazon Prime", "Audible"} <= set(found)
    assert found["Netflix"]["monthly_amount"] == 649.0
    assert 20 <= found["Netflix"]["median_gap_days"] <= 40
    assert found["Netflix"]["annualised"] == 649.0 * 12


def test_one_off_purchase_is_not_recurring(transactions):
    assert "Croma" not in {r["merchant"] for r in detect_recurring(transactions)}
