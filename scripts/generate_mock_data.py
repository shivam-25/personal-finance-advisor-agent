"""Builds the mock core-banking dataset used by the prototype.

Deterministic: a fixed seed means the committed JSON can always be regenerated
byte-for-byte, which keeps the demo and the tests reproducible.

    python scripts/generate_mock_data.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "advisor" / "data"
AS_OF = date(2026, 11, 7)
MONTHS_BACK = 6
SEED = 20261107

# merchant, category, channel, (low, high), transactions per month
Spec = tuple[str, str, str, tuple[int, int], tuple[int, int]]

COMMON: list[Spec] = [
    ("Swiggy", "Food & Dining", "UPI", (260, 820), (5, 9)),
    ("Zomato", "Food & Dining", "UPI", (240, 760), (4, 8)),
    ("Third Wave Coffee", "Food & Dining", "CARD", (180, 480), (2, 5)),
    ("Blinkit", "Groceries", "UPI", (320, 1400), (4, 7)),
    ("BigBasket", "Groceries", "CARD", (900, 3200), (1, 3)),
    ("Amazon India", "Shopping", "CARD", (450, 4200), (2, 5)),
    ("Myntra", "Shopping", "CARD", (800, 3600), (1, 3)),
    ("Uber", "Transport", "UPI", (90, 420), (6, 12)),
    ("Indian Oil", "Transport", "CARD", (1200, 2600), (1, 2)),
    ("BESCOM Electricity", "Bills & Utilities", "NETBANKING", (900, 2400), (1, 1)),
    ("Airtel Postpaid", "Bills & Utilities", "AUTOPAY", (699, 699), (1, 1)),
    ("ACT Fibernet", "Bills & Utilities", "AUTOPAY", (1180, 1180), (1, 1)),
    ("Apollo Pharmacy", "Health & Wellness", "CARD", (240, 1100), (1, 3)),
    ("Cult.fit", "Health & Wellness", "AUTOPAY", (1499, 1499), (1, 1)),
    ("PVR Cinemas", "Entertainment", "CARD", (420, 1300), (1, 3)),
    ("Netflix", "Entertainment", "AUTOPAY", (649, 649), (1, 1)),
    ("Spotify", "Entertainment", "AUTOPAY", (119, 119), (1, 1)),
    ("Amazon Prime", "Entertainment", "AUTOPAY", (299, 299), (1, 1)),
]

AUTOPAY_MERCHANTS = {"Airtel Postpaid", "ACT Fibernet", "Cult.fit", "Netflix", "Spotify", "Amazon Prime"}

CUSTOMERS = [
    {
        "customer_id": "CUST001",
        "name": "Ananya Sharma",
        "segment": "Salaried - Premium",
        "city": "Bengaluru",
        "age": 29,
        "monthly_income": 145000.0,
        "risk_profile": "moderate",
        "relationship_since": "2019-04-12",
        "rent": 34000.0,
        "sip": 20000.0,
        "travel_months": {5: (1, 2), 2: (1, 1)},
        "budgets": {
            "Food & Dining": 12000.0,
            "Shopping": 10000.0,
            "Entertainment": 4000.0,
            "Travel": 8000.0,
            "Transport": 6000.0,
            "Groceries": 12000.0,
        },
        "goals": [
            {"goal_id": "GOAL-01", "name": "Emergency fund", "target_amount": 600000.0,
             "saved_amount": 310000.0, "target_date": "2027-12-31", "priority": "high"},
            {"goal_id": "GOAL-02", "name": "Japan trip", "target_amount": 250000.0,
             "saved_amount": 48000.0, "target_date": "2027-06-30", "priority": "medium"},
        ],
        # the story we want the agents to discover in the most recent full month
        "story": {"Food & Dining": 1.32, "Shopping": 1.18, "Entertainment": 1.22},
    },
    {
        "customer_id": "CUST002",
        "name": "Rohit Verma",
        "segment": "Salaried - Classic",
        "city": "Pune",
        "age": 34,
        "monthly_income": 92000.0,
        "risk_profile": "aggressive",
        "relationship_since": "2016-09-02",
        "rent": 26000.0,
        "sip": 5000.0,
        "travel_months": {4: (1, 2)},
        "budgets": {
            "Food & Dining": 9000.0,
            "Shopping": 7000.0,
            "Entertainment": 3000.0,
            "Travel": 6000.0,
            "Transport": 5000.0,
            "Groceries": 9000.0,
        },
        "goals": [
            {"goal_id": "GOAL-11", "name": "Home down payment", "target_amount": 1500000.0,
             "saved_amount": 420000.0, "target_date": "2029-03-31", "priority": "high"},
        ],
        "story": {"Shopping": 1.55, "Entertainment": 1.4, "Food & Dining": 1.15},
    },
    {
        "customer_id": "CUST003",
        "name": "Meera Iyer",
        "segment": "Self-employed",
        "city": "Chennai",
        "age": 41,
        "monthly_income": 118000.0,
        "risk_profile": "conservative",
        "relationship_since": "2013-01-20",
        "rent": 22000.0,
        "sip": 30000.0,
        "travel_months": {3: (1, 1)},
        # budgets set comfortably above her own averages: she stays healthy
        "budgets": {
            "Food & Dining": 11000.0,
            "Shopping": 18000.0,
            "Entertainment": 4000.0,
            "Travel": 14000.0,
            "Transport": 7500.0,
            "Groceries": 14000.0,
        },
        "goals": [
            {"goal_id": "GOAL-21", "name": "Child education", "target_amount": 2500000.0,
             "saved_amount": 1180000.0, "target_date": "2032-06-30", "priority": "high"},
        ],
        "story": {},
    },
]


def month_starts(as_of: date, count: int) -> list[date]:
    """Oldest-first list of month-start dates, ending with the month of `as_of`."""
    cursor = as_of.replace(day=1)
    out = [cursor]
    for _ in range(count - 1):
        cursor = (cursor - timedelta(days=1)).replace(day=1)
        out.append(cursor)
    return list(reversed(out))


def month_end(start: date, as_of: date) -> date:
    nxt = (start.replace(day=28) + timedelta(days=7)).replace(day=1)
    return min(nxt - timedelta(days=1), as_of)


class Builder:
    def __init__(self, customer: dict, rng: random.Random) -> None:
        self.c = customer
        self.rng = rng
        self.rows: list[dict] = []
        self.seq = 0
        self.account_id = f"ACC{customer['customer_id'][-3:]}001"

    def add(self, when: date, merchant: str, desc: str, amount: float,
            direction: str, channel: str, recurring: bool = False) -> None:
        self.seq += 1
        self.rows.append({
            "txn_id": f"TXN{self.c['customer_id'][-3:]}{self.seq:05d}",
            "account_id": self.account_id,
            "posted_on": when.isoformat(),
            "description": desc,
            "merchant": merchant,
            "amount": round(amount, 2),
            "direction": direction,
            "channel": channel,
            "is_recurring": recurring,
        })

    def _billing_day(self, merchant: str, start: date, end: date) -> date | None:
        day = 2 + (sum(map(ord, merchant)) % 24)
        when = start + timedelta(days=day - 1)
        return when if when <= end else None

    def day_in(self, start: date, end: date) -> date:
        span = (end - start).days
        return start + timedelta(days=self.rng.randint(0, max(span, 0)))

    def build(self) -> list[dict]:
        months = month_starts(AS_OF, MONTHS_BACK)
        last_full = len(months) - 2  # index of the most recent complete month
        for idx, start in enumerate(months):
            end = month_end(start, AS_OF)
            partial = end.day / 30.0 if end < month_end(start, date(2099, 1, 1)) else 1.0
            self._salary(start, end)
            self._fixed(start, end, partial)
            self._discretionary(idx, last_full, start, end, partial)
            self._travel(idx, start, end)
        self._anomalies(months, last_full)
        self.rows.sort(key=lambda r: (r["posted_on"], r["txn_id"]))
        return self.rows

    def _salary(self, start: date, end: date) -> None:
        pay_day = min(start.replace(day=1) + timedelta(days=0), end)
        if pay_day > end:
            return
        self.add(pay_day, f"{self.c['segment'].split(' - ')[0]} Payroll",
                 "SALARY CREDIT", self.c["monthly_income"], "credit", "NEFT", True)

    def _fixed(self, start: date, end: date, partial: float) -> None:
        rent_day = start + timedelta(days=2)
        if rent_day <= end:
            self.add(rent_day, "Landlord Transfer", "RENT PAYMENT",
                     self.c["rent"], "debit", "NETBANKING", True)
        sip_day = start + timedelta(days=4)
        if sip_day <= end:
            self.add(sip_day, "HDFC Mutual Fund", "SIP INSTALMENT",
                     self.c["sip"], "debit", "AUTOPAY", True)

    def _discretionary(self, idx: int, last_full: int, start: date,
                       end: date, partial: float) -> None:
        story = self.c["story"] if idx == last_full else {}
        for merchant, category, channel, (lo, hi), (nlo, nhi) in COMMON:
            recurring = merchant in AUTOPAY_MERCHANTS
            count = 1 if recurring else self.rng.randint(nlo, nhi)
            if not recurring and partial < 1.0:
                count = int(round(count * partial))
            multiplier = story.get(category, 1.0)
            for _ in range(count):
                # autopay mandates hit on a stable day of the month
                when = (
                    self._billing_day(merchant, start, end)
                    if recurring
                    else self.day_in(start, end)
                )
                if when is None:
                    continue
                amount = lo if lo == hi else self.rng.uniform(lo, hi) * multiplier
                self.add(when, merchant, f"{merchant.upper()} PAYMENT",
                         amount, "debit", channel, recurring)

    def _travel(self, idx: int, start: date, end: date) -> None:
        plan = self.c["travel_months"].get(idx)
        if not plan:
            return
        for _ in range(self.rng.randint(*plan)):
            when = self.day_in(start, end)
            merchant = self.rng.choice(["MakeMyTrip", "IRCTC", "IndiGo", "OYO Rooms"])
            self.add(when, merchant, f"{merchant.upper()} BOOKING",
                     self.rng.uniform(3200, 11500), "debit", "CARD")

    def _anomalies(self, months: list[date], last_full: int) -> None:
        """Hand-placed outliers so the detection tools have something real to find."""
        start = months[last_full]
        end = month_end(start, AS_OF)
        if self.c["customer_id"] == "CUST001":
            day = start + timedelta(days=18)
            self.add(day, "Croma", "CROMA ELECTRONICS PURCHASE", 42990, "debit", "CARD")
            # duplicate charge, same merchant/amount, minutes apart
            dup_day = start + timedelta(days=9)
            self.add(dup_day, "Zomato", "ZOMATO PAYMENT", 1248.0, "debit", "UPI")
            self.add(dup_day, "Zomato", "ZOMATO PAYMENT", 1248.0, "debit", "UPI")
            # new subscription that started two months ago and keeps billing
            for k in (last_full - 1, last_full, last_full + 1):
                s = months[k]
                d = s + timedelta(days=6)
                if d <= month_end(s, AS_OF):
                    self.add(d, "Audible", "AUDIBLE MEMBERSHIP", 199, "debit", "AUTOPAY", True)
            # current-month overspend that run-rate projection should catch
            cur = months[-1]
            for _ in range(5):
                self.add(self.day_in(cur, AS_OF), "Swiggy", "SWIGGY PAYMENT",
                         self.rng.uniform(520, 980), "debit", "UPI")
            self.add(cur + timedelta(days=3), "PVR Cinemas", "PVR CINEMAS PAYMENT",
                     2480, "debit", "CARD")
        if self.c["customer_id"] == "CUST002":
            self.add(start + timedelta(days=21), "Myntra", "MYNTRA PAYMENT",
                     18760, "debit", "CARD")
            self.add(start + timedelta(days=22), "Bank Charges", "LATE PAYMENT FEE",
                     590, "debit", "SYSTEM")
            self.add(months[-1] + timedelta(days=2), "Bank Charges", "OVERLIMIT FEE",
                     450, "debit", "SYSTEM")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles, budgets, goals, transactions = [], {}, {}, {}
    for customer in CUSTOMERS:
        rng = random.Random(SEED + int(customer["customer_id"][-3:]))
        rows = Builder(customer, rng).build()
        transactions[customer["customer_id"]] = rows
        budgets[customer["customer_id"]] = customer["budgets"]
        goals[customer["customer_id"]] = customer["goals"]
        profiles.append({
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "segment": customer["segment"],
            "city": customer["city"],
            "age": customer["age"],
            "monthly_income": customer["monthly_income"],
            "risk_profile": customer["risk_profile"],
            "relationship_since": customer["relationship_since"],
            "accounts": [{
                "account_id": f"ACC{customer['customer_id'][-3:]}001",
                "type": "SAVINGS",
                "currency": "INR",
                "balance": round(customer["monthly_income"] * rng.uniform(1.4, 3.1), 2),
            }],
        })

    payload = {
        "as_of": AS_OF.isoformat(),
        "generated_with_seed": SEED,
        "currency": "INR",
        "profiles": profiles,
        "budgets": budgets,
        "goals": goals,
        "transactions": transactions,
    }
    out = OUT_DIR / "bank_dataset.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in transactions.values())
    print(f"wrote {out.relative_to(ROOT)}  customers={len(profiles)} transactions={total}")


if __name__ == "__main__":
    main()
