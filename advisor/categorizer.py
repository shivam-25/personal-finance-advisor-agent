"""Merchant enrichment: a bank would call a vendor here; this is a rule engine."""

from __future__ import annotations

import re
from collections import Counter, defaultdict


from .domain import Direction, Transaction

MERCHANT_MAP: dict[str, str] = {
    "swiggy": "Food & Dining",
    "zomato": "Food & Dining",
    "third wave coffee": "Food & Dining",
    "dominos": "Food & Dining",
    "blinkit": "Groceries",
    "bigbasket": "Groceries",
    "zepto": "Groceries",
    "dmart": "Groceries",
    "amazon india": "Shopping",
    "myntra": "Shopping",
    "croma": "Shopping",
    "flipkart": "Shopping",
    "uber": "Transport",
    "ola": "Transport",
    "indian oil": "Transport",
    "makemytrip": "Travel",
    "irctc": "Travel",
    "indigo": "Travel",
    "oyo rooms": "Travel",
    "bescom electricity": "Bills & Utilities",
    "airtel postpaid": "Bills & Utilities",
    "act fibernet": "Bills & Utilities",
    "apollo pharmacy": "Health & Wellness",
    "cult.fit": "Health & Wellness",
    "pvr cinemas": "Entertainment",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "amazon prime": "Entertainment",
    "audible": "Entertainment",
    "landlord transfer": "Rent",
    "hdfc mutual fund": "Investments",
    "bank charges": "Fees & Charges",
}

KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    (r"\bsalary\b|\bpayroll\b", "Income"),
    (r"\brent\b", "Rent"),
    (r"\bsip\b|mutual fund|\belss\b", "Investments"),
    # domain rules run before the generic fee rule, so "tuition fee" is Education
    (r"tuition|school|college|course|udemy|coursera", "Education"),
    (r"pharmac|hospital|clinic|diagnost|doctor", "Health & Wellness"),
    (r"electric|broadband|postpaid|gas bill|water bill", "Bills & Utilities"),
    (r"late payment|bank charge|overlimit|penalty|\bfee\b|\bcharges?\b", "Fees & Charges"),
    (r"fuel|petrol|diesel|metro|cab", "Transport"),
    (r"flight|hotel|booking|travel", "Travel"),
    (r"movie|cinema|subscription|music", "Entertainment"),
    (r"grocer|supermarket", "Groceries"),
    (r"restaurant|food|cafe|coffee|dining", "Food & Dining"),
    (r"upi/p2p|imps|neft|transfer", "Transfers"),
)

CHANNEL_HINTS = {"AUTOPAY": "Bills & Utilities", "SYSTEM": "Fees & Charges"}


def categorize(txn: Transaction) -> Transaction:
    """Merchant match, then keywords, then channel hint, then fallback."""
    if txn.direction is Direction.CREDIT and "salary" in txn.description.lower():
        txn.category, txn.category_confidence = "Income", 0.99
        return txn

    merchant = txn.merchant.lower().strip()
    if merchant in MERCHANT_MAP:
        txn.category, txn.category_confidence = MERCHANT_MAP[merchant], 0.97
        return txn

    for name, category in MERCHANT_MAP.items():
        if name in merchant:
            txn.category, txn.category_confidence = category, 0.9
            return txn

    haystack = f"{txn.merchant} {txn.description}".lower()
    for pattern, category in KEYWORD_RULES:
        if re.search(pattern, haystack):
            txn.category, txn.category_confidence = category, 0.75
            return txn

    if txn.channel in CHANNEL_HINTS:
        txn.category, txn.category_confidence = CHANNEL_HINTS[txn.channel], 0.45
        return txn

    txn.category, txn.category_confidence = "Uncategorised", 0.2
    return txn


def categorize_all(transactions: list[Transaction]) -> list[Transaction]:
    return [categorize(t) for t in transactions]


def coverage(transactions: list[Transaction]) -> dict[str, float]:
    """Share of the book the rule engine is confident about."""
    if not transactions:
        return {"categorised_pct": 0.0, "mean_confidence": 0.0, "low_confidence": 0}
    low = [t for t in transactions if t.category_confidence < 0.5]
    known = [t for t in transactions if t.category != "Uncategorised"]
    return {
        "categorised_pct": round(100 * len(known) / len(transactions), 1),
        "mean_confidence": round(
            sum(t.category_confidence for t in transactions) / len(transactions), 3
        ),
        "low_confidence": len(low),
    }


def detect_recurring(
    transactions: list[Transaction], min_occurrences: int = 3, tolerance: float = 0.12
) -> list[dict]:
    """Subscription-like debits: same merchant, stable amount, monthly cadence."""
    buckets: dict[str, list[Transaction]] = defaultdict(list)
    for txn in transactions:
        if txn.direction is Direction.DEBIT:
            buckets[txn.merchant].append(txn)

    found = []
    for merchant, rows in buckets.items():
        rows.sort(key=lambda t: t.posted_on)
        amounts = Counter(round(t.amount, 0) for t in rows)
        amount, hits = amounts.most_common(1)[0]
        if hits < min_occurrences:
            continue
        stable = [t for t in rows if abs(t.amount - amount) <= max(amount * tolerance, 1)]
        if len(stable) < min_occurrences:
            continue
        months = sorted({t.month for t in stable})
        if len(months) < min_occurrences:
            continue
        gaps = [(b.posted_on - a.posted_on).days for a, b in zip(stable, stable[1:])]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
        if not 20 <= median_gap <= 40:
            continue
        found.append({
            "merchant": merchant,
            "category": stable[-1].category,
            "monthly_amount": round(float(amount), 2),
            "occurrences": len(stable),
            "months_billed": len(months),
            "median_gap_days": median_gap,
            "first_seen": stable[0].posted_on.isoformat(),
            "last_seen": stable[-1].posted_on.isoformat(),
            "channel": stable[-1].channel,
            "annualised": round(float(amount) * 12, 2),
        })
    return sorted(found, key=lambda r: r["monthly_amount"], reverse=True)
