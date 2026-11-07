"""Deterministic statistics. No agent logic here, so every number is testable."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Iterable

from .domain import DISCRETIONARY, ESSENTIAL, NON_SPEND, Direction, Transaction


def month_key(value: date) -> str:
    return value.strftime("%Y-%m")


def month_bounds(key: str) -> tuple[date, date]:
    year, month = (int(p) for p in key.split("-"))
    start = date(year, month, 1)
    end = (start.replace(day=28) + timedelta(days=7)).replace(day=1) - timedelta(days=1)
    return start, end


def shift_month(key: str, delta: int) -> str:
    year, month = (int(p) for p in key.split("-"))
    index = year * 12 + (month - 1) + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def spend_rows(transactions: Iterable[Transaction]) -> list[Transaction]:
    """Debits that are real consumption, excluding SIPs and transfers."""
    return [
        t
        for t in transactions
        if t.direction is Direction.DEBIT and t.category not in NON_SPEND
    ]


def total_spend(transactions: Iterable[Transaction]) -> float:
    return round(sum(t.amount for t in spend_rows(transactions)), 2)


def total_income(transactions: Iterable[Transaction]) -> float:
    return round(
        sum(t.amount for t in transactions if t.direction is Direction.CREDIT), 2
    )


def by_category(transactions: Iterable[Transaction]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Transaction]] = defaultdict(list)
    for txn in spend_rows(transactions):
        buckets[txn.category].append(txn)
    grand = sum(t.amount for rows in buckets.values() for t in rows) or 1.0
    out = {}
    for category, rows in buckets.items():
        amount = sum(t.amount for t in rows)
        out[category] = {
            "total": round(amount, 2),
            "txn_count": len(rows),
            "share_pct": round(100 * amount / grand, 1),
            "average_txn": round(amount / len(rows), 2),
            "largest_txn": round(max(t.amount for t in rows), 2),
            "top_merchant": _top_merchant(rows),
            "type": _category_type(category),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["total"], reverse=True))


def _top_merchant(rows: list[Transaction]) -> dict[str, Any]:
    tally: dict[str, float] = defaultdict(float)
    for txn in rows:
        tally[txn.merchant] += txn.amount
    merchant, amount = max(tally.items(), key=lambda kv: kv[1])
    return {"merchant": merchant, "total": round(amount, 2)}


def _category_type(category: str) -> str:
    if category in DISCRETIONARY:
        return "discretionary"
    if category in ESSENTIAL:
        return "essential"
    return "other"


def by_month(transactions: Iterable[Transaction]) -> dict[str, float]:
    tally: dict[str, float] = defaultdict(float)
    for txn in spend_rows(transactions):
        tally[txn.month] += txn.amount
    return {k: round(v, 2) for k, v in sorted(tally.items())}


def category_by_month(transactions: Iterable[Transaction]) -> dict[str, dict[str, float]]:
    tally: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for txn in spend_rows(transactions):
        tally[txn.category][txn.month] += txn.amount
    return {
        cat: {m: round(v, 2) for m, v in sorted(months.items())}
        for cat, months in tally.items()
    }


def compare_months(
    transactions: Iterable[Transaction], current: str, previous: str
) -> dict[str, Any]:
    """Month-over-month delta per category, biggest movers first."""
    series = category_by_month(transactions)
    rows = []
    for category, months in series.items():
        now, before = months.get(current, 0.0), months.get(previous, 0.0)
        if not now and not before:
            continue
        delta = now - before
        change_pct = None if before == 0 else round(100 * delta / before, 1)
        rows.append({
            "category": category,
            "current": round(now, 2),
            "previous": round(before, 2),
            "delta": round(delta, 2),
            "change_pct": change_pct,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        })
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    totals = by_month(transactions)
    now_total, prev_total = totals.get(current, 0.0), totals.get(previous, 0.0)
    return {
        "current_month": current,
        "previous_month": previous,
        "current_total": round(now_total, 2),
        "previous_total": round(prev_total, 2),
        "total_change_pct": (
            None if prev_total == 0 else round(100 * (now_total - prev_total) / prev_total, 1)
        ),
        "movers": rows,
    }


def amount_outliers(
    transactions: Iterable[Transaction], z_threshold: float = 2.5, min_sample: int = 6
) -> list[dict[str, Any]]:
    """Per-category z-score outliers on debit amounts."""
    buckets: dict[str, list[Transaction]] = defaultdict(list)
    for txn in spend_rows(transactions):
        buckets[txn.category].append(txn)

    found = []
    for category, rows in buckets.items():
        if len(rows) < min_sample:
            continue
        amounts = [t.amount for t in rows]
        mean = statistics.fmean(amounts)
        spread = statistics.pstdev(amounts)
        if spread < 1e-6:
            continue
        for txn in rows:
            z = (txn.amount - mean) / spread
            if z >= z_threshold:
                found.append({
                    "type": "amount_outlier",
                    "txn_id": txn.txn_id,
                    "posted_on": txn.posted_on.isoformat(),
                    "merchant": txn.merchant,
                    "category": category,
                    "amount": round(txn.amount, 2),
                    "category_mean": round(mean, 2),
                    "z_score": round(z, 2),
                })
    return sorted(found, key=lambda r: r["z_score"], reverse=True)


def duplicate_charges(
    transactions: Iterable[Transaction], window_days: int = 1
) -> list[dict[str, Any]]:
    """Same merchant and amount inside a short window."""
    buckets: dict[tuple[str, float], list[Transaction]] = defaultdict(list)
    for txn in spend_rows(transactions):
        buckets[(txn.merchant, round(txn.amount, 2))].append(txn)

    found = []
    for (merchant, amount), rows in buckets.items():
        rows.sort(key=lambda t: t.posted_on)
        for first, second in zip(rows, rows[1:]):
            if (second.posted_on - first.posted_on).days <= window_days:
                found.append({
                    "type": "duplicate_charge",
                    "merchant": merchant,
                    "category": first.category,
                    "amount": amount,
                    "posted_on": second.posted_on.isoformat(),
                    "txn_ids": [first.txn_id, second.txn_id],
                })
    return found


def category_spikes(
    transactions: Iterable[Transaction],
    current: str,
    lookback: int = 3,
    threshold_pct: float = 20.0,
    min_amount: float = 1500.0,
) -> list[dict[str, Any]]:
    """Categories running above their own trailing baseline."""
    series = category_by_month(transactions)
    months = [shift_month(current, -i) for i in range(1, lookback + 1)]
    found = []
    for category, values in series.items():
        now = values.get(current, 0.0)
        baseline_values = [values.get(m, 0.0) for m in months if m in values]
        if not baseline_values or now < min_amount:
            continue
        baseline = statistics.fmean(baseline_values)
        if baseline <= 0:
            continue
        change = 100 * (now - baseline) / baseline
        if change >= threshold_pct:
            found.append({
                "type": "category_spike",
                "category": category,
                "current": round(now, 2),
                "baseline": round(baseline, 2),
                "baseline_months": [m for m in months if m in values],
                "change_pct": round(change, 1),
                "excess": round(now - baseline, 2),
            })
    return sorted(found, key=lambda r: r["excess"], reverse=True)


def budget_status(
    transactions: Iterable[Transaction],
    budgets: dict[str, float],
    month: str,
    as_of: date,
) -> list[dict[str, Any]]:
    """Utilisation plus a projection to month end."""
    start, end = month_bounds(month)
    elapsed = max((min(as_of, end) - start).days + 1, 1)
    total_days = (end - start).days + 1
    partial = as_of < end
    series = category_by_month(transactions)

    # trailing rate damps a projection made only days into the month
    history = [shift_month(month, -i) for i in range(1, 4)]
    rows = []
    for category, limit in budgets.items():
        months = series.get(category, {})
        spent = months.get(month, 0.0)
        past = [months[m] for m in history if m in months]
        used_pct = round(100 * spent / limit, 1) if limit else 0.0

        if partial:
            month_rate = spent / elapsed
            trailing_rate = (statistics.fmean(past) / 30.0) if past else month_rate
            weight = elapsed / total_days
            blended = weight * month_rate + (1 - weight) * trailing_rate
            projected = round(spent + blended * (total_days - elapsed), 2)
            confidence = "low" if elapsed < 7 else ("medium" if elapsed < 15 else "high")
        else:
            projected = round(spent, 2)
            confidence = "actual"

        projected_pct = round(100 * projected / limit, 1) if limit else 0.0
        if used_pct >= 100:
            state = "breached"
        elif projected_pct >= 100:
            state = "on_track_to_breach"
        elif used_pct >= 80:
            state = "near_limit"
        else:
            state = "healthy"
        rows.append({
            "category": category,
            "budget": round(limit, 2),
            "spent": round(spent, 2),
            "remaining": round(limit - spent, 2),
            "used_pct": used_pct,
            "projected_spend": projected,
            "projected_pct": projected_pct,
            "projection_confidence": confidence,
            "days_elapsed": elapsed,
            "days_in_month": total_days,
            "state": state,
        })
    order = {"breached": 0, "on_track_to_breach": 1, "near_limit": 2, "healthy": 3}
    rows.sort(key=lambda r: (order[r["state"]], -r["used_pct"]))
    return rows


def savings_capacity(
    transactions: Iterable[Transaction], month: str, monthly_income: float
) -> dict[str, Any]:
    """Split a month into essential, discretionary and invested money."""
    rows = [t for t in transactions if t.month == month]
    spend = spend_rows(rows)
    essential = sum(t.amount for t in spend if t.category in ESSENTIAL)
    discretionary = sum(t.amount for t in spend if t.category in DISCRETIONARY)
    other = sum(t.amount for t in spend) - essential - discretionary
    invested = sum(
        t.amount
        for t in rows
        if t.direction is Direction.DEBIT and t.category == "Investments"
    )
    income = total_income(rows) or monthly_income
    outflow = essential + discretionary + other
    surplus = income - outflow - invested
    return {
        "month": month,
        "income": round(income, 2),
        "essential_spend": round(essential, 2),
        "discretionary_spend": round(discretionary, 2),
        "other_spend": round(other, 2),
        "invested": round(invested, 2),
        "total_outflow": round(outflow + invested, 2),
        "surplus": round(surplus, 2),
        "savings_rate_pct": round(100 * (surplus + invested) / income, 1) if income else 0.0,
        "discretionary_share_pct": round(100 * discretionary / income, 1) if income else 0.0,
    }


def simulate_reductions(
    baseline: dict[str, float], reductions: dict[str, float]
) -> dict[str, Any]:
    """Apply percentage cuts to category baselines."""
    lines = []
    for category, cut_pct in reductions.items():
        current = baseline.get(category, 0.0)
        saved = current * cut_pct / 100
        lines.append({
            "category": category,
            "current_monthly": round(current, 2),
            "reduction_pct": cut_pct,
            "new_monthly": round(current - saved, 2),
            "monthly_saving": round(saved, 2),
        })
    monthly = sum(line["monthly_saving"] for line in lines)
    return {
        "lines": sorted(lines, key=lambda r: r["monthly_saving"], reverse=True),
        "monthly_saving": round(monthly, 2),
        "annual_saving": round(monthly * 12, 2),
    }


def goal_projection(
    goal: dict[str, Any], monthly_contribution: float, as_of: date
) -> dict[str, Any]:
    target = float(goal["target_amount"])
    saved = float(goal["saved_amount"])
    gap = max(target - saved, 0.0)
    target_date = date.fromisoformat(goal["target_date"])
    months_left = max(
        (target_date.year - as_of.year) * 12 + target_date.month - as_of.month, 0
    )
    required = round(gap / months_left, 2) if months_left else gap
    months_needed = (
        None if monthly_contribution <= 0 else int(-(-gap // monthly_contribution))
    )
    return {
        "goal_id": goal["goal_id"],
        "name": goal["name"],
        "target_amount": round(target, 2),
        "saved_amount": round(saved, 2),
        "progress_pct": round(100 * saved / target, 1) if target else 0.0,
        "gap": round(gap, 2),
        "target_date": goal["target_date"],
        "months_remaining": months_left,
        "required_monthly": required,
        "months_at_current_rate": months_needed,
        "on_track": monthly_contribution >= required if months_left else saved >= target,
    }
