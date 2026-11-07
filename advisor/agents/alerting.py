"""Alerting Agent.

Decides what the customer must be told now rather than what is merely
interesting: budget utilisation, month-end projections and anomalies become
severity-ranked alerts. Acknowledged alerts are suppressed, and projections
made a few days into a month are demoted.
"""

from __future__ import annotations

from ..domain import Alert, Severity, money, pct
from .base import Agent, AgentRequest, AgentResponse

NEAR_LIMIT_PCT = 80.0


class AlertingAgent(Agent):
    name = "alerting"
    description = (
        "Monitors budget thresholds, projects month-end overspend and raises "
        "severity-ranked proactive alerts, suppressing acknowledged ones."
    )
    capabilities = ("alert.budget", "alert.projection", "alert.anomaly")
    tasks = ("check_alerts",)

    def handle(self, request: AgentRequest) -> AgentResponse:
        response = AgentResponse(agent=self.name, task=request.task)
        month = request.params.get("month")

        status = self._call(response, "get_budget_status", {"month": month} if month else {})
        if status is None:
            response.status = "failed"
            response.summary = "Budget data unavailable."
            return response

        # An acknowledgement key is either an alert id or a whole category.
        acknowledged = {
            note.key for note in self.memory.profile.all("acknowledged_alert")
        }

        partial = status["is_partial_month"]
        response.reasoning.append(
            f"Evaluating {len(status['budgets'])} budgets for {status['month']} "
            + ("(month in progress, projections applied)" if partial else "(closed month)")
        )

        for row in status["budgets"]:
            alert = self._budget_alert(row, status, partial)
            if alert:
                response.alerts.append(alert)

        self._anomaly_alerts(request, response)

        if acknowledged:
            kept = [
                a for a in response.alerts
                if a.alert_id not in acknowledged and a.category not in acknowledged
            ]
            response.reasoning.append(
                f"Suppressed {len(response.alerts) - len(kept)} acknowledged alert(s) "
                f"matching: {', '.join(sorted(acknowledged))}"
            )
            response.alerts = kept

        response.alerts.sort(key=lambda a: a.severity.rank, reverse=True)
        for alert in response.alerts:
            self._finding(
                response,
                "alert",
                alert.title,
                alert.message,
                alert.severity,
                alert.evidence,
                ("alert", alert.triggered_by),
            )

        response.data = {
            "month": status["month"],
            "budget_status": status,
            "alert_count": len(response.alerts),
            "highest_severity": (
                response.alerts[0].severity.value if response.alerts else "none"
            ),
        }
        response.summary = (
            f"{len(response.alerts)} active alert(s) for {status['month']}."
            if response.alerts
            else f"No threshold breaches for {status['month']}."
        )
        self.memory.working.put("alerts", response.data)
        return response

    # ------------------------------------------------------------------
    def _budget_alert(self, row: dict, status: dict, partial: bool) -> Alert | None:
        category = row["category"]
        confidence = row.get("projection_confidence", "actual")

        if row["state"] == "breached":
            over = -row["remaining"]
            return Alert(
                alert_id=f"ALERT-{category[:3].upper()}-BREACH",
                title=f"{category} budget exceeded",
                message=(
                    f"You have spent {money(row['spent'])} against a "
                    f"{money(row['budget'])} {category} budget - "
                    f"{money(over)} over, {row['used_pct']}% used."
                ),
                severity=Severity.CRITICAL if row["used_pct"] >= 120 else Severity.HIGH,
                category=category,
                triggered_by="budget_breach",
                evidence=row,
            )

        if row["state"] == "on_track_to_breach" and partial:
            severity = Severity.MEDIUM if confidence in ("high", "medium") else Severity.LOW
            return Alert(
                alert_id=f"ALERT-{category[:3].upper()}-PROJ",
                title=f"{category} on track to breach budget",
                message=(
                    f"{money(row['spent'])} of {money(row['budget'])} used "
                    f"{row['days_elapsed']} days in. At the current rate the month "
                    f"closes near {money(row['projected_spend'])} "
                    f"({row['projected_pct']}% of budget). "
                    f"Projection confidence: {confidence}."
                ),
                severity=severity,
                category=category,
                triggered_by="projection",
                evidence=row,
            )

        if row["used_pct"] >= NEAR_LIMIT_PCT:
            return Alert(
                alert_id=f"ALERT-{category[:3].upper()}-NEAR",
                title=f"Nearing your {category} budget limit",
                message=(
                    f"{row['used_pct']}% of the {money(row['budget'])} {category} "
                    f"budget is used, {money(row['remaining'])} remaining."
                ),
                severity=Severity.MEDIUM,
                category=category,
                triggered_by="near_limit",
                evidence=row,
            )
        return None

    def _anomaly_alerts(self, request: AgentRequest, response: AgentResponse) -> None:
        """Reuse the analyst's anomaly evidence, or fetch it."""
        upstream = request.evidence.get("spending_analysis") or {}
        anomalies = (upstream.get("anomalies") or {}) if isinstance(upstream, dict) else {}
        if not anomalies:
            anomalies = self._call(response, "detect_spending_anomalies", {}) or {}
            response.reasoning.append(
                "No upstream anomaly evidence on the blackboard; fetched it directly."
            )
        else:
            response.reasoning.append(
                "Reused anomaly evidence already produced by the spending agent."
            )

        for dup in anomalies.get("duplicate_charges", []):
            response.alerts.append(Alert(
                alert_id=f"ALERT-DUP-{dup['merchant'][:4].upper()}",
                title=f"Duplicate charge at {dup['merchant']}",
                message=(
                    f"Two identical {money(dup['amount'])} debits to "
                    f"{dup['merchant']} on {dup['posted_on']}. "
                    "You can raise a dispute from the app if this was not intended."
                ),
                severity=Severity.HIGH,
                category=dup.get("category", "Uncategorised"),
                triggered_by="duplicate_charge",
                evidence=dup,
            ))

        spike_month = anomalies.get("month", "the latest month")
        for spike in anomalies.get("category_spikes", [])[:2]:
            if spike["change_pct"] < 40:
                continue
            response.alerts.append(Alert(
                alert_id=f"ALERT-SPIKE-{spike['category'][:3].upper()}",
                title=f"{spike['category']} spending spike in {spike_month}",
                message=(
                    f"{money(spike['current'])} in {spike['category']} during "
                    f"{spike_month}, {pct(spike['change_pct'])} above your "
                    f"{money(spike['baseline'])} three-month average."
                ),
                severity=Severity.MEDIUM,
                category=spike["category"],
                triggered_by="spike",
                evidence=spike,
            ))
