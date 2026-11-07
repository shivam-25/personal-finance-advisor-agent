"""Spending Analysis Agent.

Turns the transaction book into quantified evidence the other agents consume.
Pipeline: enrich, aggregate, compare to the previous month, detect anomalies,
then separate habitual drift from one-off purchases.
"""

from __future__ import annotations

from typing import Any

from ..domain import DISCRETIONARY, Severity, money, pct
from .base import Agent, AgentRequest, AgentResponse

#: above this share of a category's month, one transaction is a one-off, not a habit
ONE_OFF_SHARE = 0.55


class SpendingAnalysisAgent(Agent):
    name = "spending_analysis"
    description = (
        "Categorises transactions, computes monthly and per-category trends, "
        "and separates habitual drift from one-off purchases."
    )
    capabilities = (
        "spend.summary",
        "spend.trend",
        "spend.comparison",
        "spend.anomalies",
    )
    tasks = ("analyze_spending", "analyze_category", "detect_anomalies")

    def handle(self, request: AgentRequest) -> AgentResponse:
        response = AgentResponse(agent=self.name, task=request.task)
        if request.task == "detect_anomalies":
            return self._anomalies_only(request, response)
        if request.task == "analyze_category":
            return self._category_deep_dive(request, response)
        return self._full_analysis(request, response)

    # ------------------------------------------------------------------
    def _full_analysis(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        month = request.params.get("month")
        baseline = request.params.get("baseline")

        enrichment = self._call(response, "categorize_transactions", {"month": month} if month else {})
        if enrichment:
            cov = enrichment["coverage"]
            response.reasoning.append(
                f"Enriched {enrichment['transaction_count']} transactions; "
                f"{cov['categorised_pct']}% categorised at mean confidence {cov['mean_confidence']}."
            )
            if cov["categorised_pct"] < 90:
                response.confidence = 0.75
                response.reasoning.append(
                    "Categorisation coverage below 90%, so category totals are indicative."
                )

        summary = self._call(response, "summarize_spend_by_category", {"month": month} if month else {})
        if summary is None:
            response.status = "failed"
            response.summary = "Could not read the transaction book for the requested period."
            return response

        month = summary["month"]
        comparison = self._call(
            response,
            "compare_periods",
            {"month": month, **({"baseline": baseline} if baseline else {})},
        )
        anomalies = self._call(response, "detect_spending_anomalies", {"month": month})
        trend = self._call(response, "get_spend_trend", {"months": 6})

        one_offs = self._classify_one_offs(summary, anomalies)
        response.reasoning.append(
            f"{month} spend {money(summary['total_spend'])} across "
            f"{len(summary['categories'])} categories."
        )

        self._finding(
            response,
            "spend_summary",
            f"{month} total spend {money(summary['total_spend'])}",
            self._top_lines(summary),
            Severity.INFO,
            {"month": month, "total_spend": summary["total_spend"],
             "categories": summary["categories"]},
            ("summary",),
        )

        if comparison:
            self._compare_findings(response, comparison, one_offs)
        if anomalies:
            self._anomaly_findings(response, anomalies, one_offs)

        response.data = {
            "month": month,
            "summary": summary,
            "comparison": comparison,
            "anomalies": anomalies,
            "trend": trend,
            "one_offs": one_offs,
            "behavioural_increases": [
                m for m in (comparison or {}).get("movers", [])
                if m["direction"] == "up"
                and m["category"] in DISCRETIONARY
                and m["category"] not in one_offs
            ],
        }
        response.summary = self._narrate(month, summary, comparison, anomalies)
        self.memory.working.put("spend.analysis", response.data)
        return response

    # ------------------------------------------------------------------
    def _classify_one_offs(self, summary: dict, anomalies: dict | None) -> dict[str, dict]:
        """Categories whose month was dominated by one large transaction.

        Recommending a cut because of a single fridge purchase is bad advice.
        """
        out: dict[str, dict] = {}
        outliers = {o["category"]: o for o in (anomalies or {}).get("amount_outliers", [])}
        for category, info in summary["categories"].items():
            share = info["largest_txn"] / info["total"] if info["total"] else 0
            if share >= ONE_OFF_SHARE and category in outliers:
                out[category] = {
                    "largest_txn": info["largest_txn"],
                    "share_of_category": round(100 * share, 1),
                    "merchant": outliers[category]["merchant"],
                    "txn_id": outliers[category]["txn_id"],
                }
        return out

    def _compare_findings(self, response: AgentResponse, comparison: dict, one_offs: dict) -> None:
        total_change = comparison["total_change_pct"]
        if total_change is not None and abs(total_change) >= 10:
            direction = "higher" if total_change > 0 else "lower"
            self._finding(
                response,
                "trend",
                f"Total spend {pct(total_change)} vs {comparison['previous_month']}",
                f"{money(comparison['current_total'])} this period against "
                f"{money(comparison['previous_total'])} previously - {direction}.",
                Severity.MEDIUM if total_change >= 20 else Severity.LOW,
                {"change_pct": total_change,
                 "current": comparison["current_total"],
                 "previous": comparison["previous_total"]},
                ("trend",),
            )

        for mover in comparison["movers"][:5]:
            if mover["direction"] != "up" or mover["change_pct"] is None:
                continue
            if mover["change_pct"] < 20 or mover["delta"] < 1000:
                continue
            category = mover["category"]
            one_off = one_offs.get(category)
            if one_off:
                detail = (
                    f"{money(mover['previous'])} -> {money(mover['current'])}, but "
                    f"{one_off['share_of_category']}% of that is a single "
                    f"{money(one_off['largest_txn'])} charge at {one_off['merchant']}. "
                    "Treated as a one-off, not a change in habit."
                )
                severity = Severity.LOW
                tags = ("mover", "one_off")
            else:
                detail = (
                    f"{money(mover['previous'])} -> {money(mover['current'])} "
                    f"({pct(mover['change_pct'])}), a sustained increase of "
                    f"{money(mover['delta'])}."
                )
                severity = Severity.HIGH if mover["change_pct"] >= 50 else Severity.MEDIUM
                tags = ("mover", "behavioural")
            self._finding(
                response,
                "category_increase",
                f"{category} up {pct(mover['change_pct'])}",
                detail,
                severity,
                {**mover, "one_off": one_off},
                tags,
            )

    def _anomaly_findings(self, response: AgentResponse, anomalies: dict, one_offs: dict) -> None:
        for dup in anomalies.get("duplicate_charges", []):
            self._finding(
                response,
                "duplicate_charge",
                f"Possible duplicate charge at {dup['merchant']}",
                f"Two {money(dup['amount'])} debits to {dup['merchant']} on "
                f"{dup['posted_on']}. Worth raising a dispute if unintended.",
                Severity.HIGH,
                dup,
                ("anomaly", "actionable"),
            )
        for outlier in anomalies.get("amount_outliers", [])[:3]:
            self._finding(
                response,
                "amount_outlier",
                f"Unusually large {outlier['category']} transaction",
                f"{money(outlier['amount'])} at {outlier['merchant']} on "
                f"{outlier['posted_on']} - {outlier['z_score']} standard deviations "
                f"above the {money(outlier['category_mean'])} category average.",
                Severity.MEDIUM,
                outlier,
                ("anomaly",),
            )

    def _anomalies_only(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        month = request.params.get("month")
        anomalies = self._call(response, "detect_spending_anomalies", {"month": month} if month else {})
        if anomalies is None:
            response.status = "failed"
            return response
        summary = self._call(response, "summarize_spend_by_category", {"month": anomalies["month"]}) or {
            "categories": {}
        }
        one_offs = self._classify_one_offs(summary, anomalies)
        self._anomaly_findings(response, anomalies, one_offs)
        for spike in anomalies.get("category_spikes", [])[:3]:
            self._finding(
                response,
                "category_spike",
                f"{spike['category']} {pct(spike['change_pct'])} above its own baseline",
                f"{money(spike['current'])} this month against a "
                f"{money(spike['baseline'])} three-month average "
                f"({money(spike['excess'])} extra).",
                Severity.MEDIUM,
                spike,
                ("anomaly", "spike"),
            )
        response.data = {"anomalies": anomalies, "one_offs": one_offs}
        response.summary = (
            f"{anomalies['anomaly_count']} anomaly signal(s) in {anomalies['month']}."
        )
        response.reasoning.append(
            f"Ran z-score, duplicate and spike detection over {anomalies['month']}."
        )
        self.memory.working.put("spend.anomalies", response.data)
        return response

    def _category_deep_dive(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        category = request.params.get("category")
        if not category:
            response.status = "failed"
            response.unmet_needs = ["category"]
            response.summary = "No category was specified for the deep dive."
            return response
        month = request.params.get("month")
        trend = self._call(response, "get_spend_trend", {"months": 6}) or {"by_category": {}}
        series = trend["by_category"].get(category, {})
        detail = self._call(
            response,
            "fetch_transactions",
            {"category": category, **({"start": f"{month}-01"} if month else {})},
        ) or {"transactions": [], "count": 0}

        merchants: dict[str, float] = {}
        for txn in detail["transactions"]:
            merchants[txn["merchant"]] = merchants.get(txn["merchant"], 0.0) + txn["amount"]
        top = sorted(merchants.items(), key=lambda kv: kv[1], reverse=True)[:5]

        months = sorted(series)
        latest = series.get(months[-1], 0.0) if months else 0.0
        average = sum(series.values()) / len(series) if series else 0.0
        response.reasoning.append(
            f"Pulled {detail['count']} {category} transactions across {len(months)} months."
        )
        self._finding(
            response,
            "category_detail",
            f"{category}: {money(latest)} latest month vs {money(average)} average",
            "Top merchants - "
            + ", ".join(f"{m} {money(v)}" for m, v in top),
            Severity.INFO,
            {"category": category, "series": series, "top_merchants": top},
            ("deep_dive",),
        )
        response.data = {
            "category": category,
            "series": series,
            "top_merchants": [{"merchant": m, "total": round(v, 2)} for m, v in top],
            "transaction_count": detail["count"],
        }
        response.summary = f"{category} detail across {len(months)} months."
        self.memory.working.put(f"spend.category.{category}", response.data)
        return response

    # ------------------------------------------------------------------
    def _top_lines(self, summary: dict) -> str:
        rows = list(summary["categories"].items())[:5]
        return "; ".join(
            f"{cat} {money(info['total'])} ({info['share_pct']}%)" for cat, info in rows
        )

    def _narrate(self, month: str, summary: dict, comparison: dict | None, anomalies: dict | None) -> str:
        parts = [f"{month}: {money(summary['total_spend'])} spent."]
        if comparison and comparison["total_change_pct"] is not None:
            parts.append(
                f"That is {pct(comparison['total_change_pct'])} against {comparison['previous_month']}."
            )
        if anomalies and anomalies["anomaly_count"]:
            parts.append(f"{anomalies['anomaly_count']} anomaly signal(s) detected.")
        return " ".join(parts)
