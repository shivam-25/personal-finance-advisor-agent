"""Recommendation Agent.

Turns evidence into a few actionable steps, each with a rupee figure from the
simulation tool. Three rules: never advise without evidence (declare
`unmet_needs` instead), never touch a protected or previously declined
category, and never bill a one-off purchase as a habit.
"""

from __future__ import annotations

from ..domain import DISCRETIONARY, Recommendation, Severity, money, pct
from .base import Agent, AgentRequest, AgentResponse

#: growth floor -> how deep a cut the evidence justifies
CUT_LADDER = ((50.0, 20.0), (25.0, 15.0), (0.0, 10.0))
MIN_MONTHLY_IMPACT = 300.0


class RecommendationAgent(Agent):
    name = "recommendation"
    description = (
        "Generates ranked, quantified savings actions from spending evidence, "
        "respecting protected categories and previously declined advice."
    )
    capabilities = ("advice.savings", "advice.budget", "advice.subscriptions", "advice.goals")
    tasks = ("recommend_savings", "recommend_budget_fix", "audit_subscriptions", "review_goals")

    def handle(self, request: AgentRequest) -> AgentResponse:
        response = AgentResponse(agent=self.name, task=request.task)

        constraints = self.memory.profile.constraints()
        protected = set(constraints["protected_categories"])
        declined = set(constraints["declined"])
        if protected or declined:
            response.reasoning.append(
                f"Memory constraints applied - protected: {sorted(protected) or 'none'}; "
                f"previously declined: {sorted(declined) or 'none'}."
            )

        if request.task == "audit_subscriptions":
            return self._subscriptions(response, protected, declined)
        if request.task == "review_goals":
            return self._goals(request, response)

        analysis = request.evidence.get("spending_analysis")
        if not analysis:
            response.status = "partial"
            response.unmet_needs = ["spend.summary"]
            response.reasoning.append(
                "No spending evidence on the blackboard - requesting spend.summary "
                "from the planner before advising."
            )
            response.summary = "Waiting on spending analysis before recommending."
            return response

        return self._savings(request, response, analysis, protected, declined)

    # ------------------------------------------------------------------
    def _savings(
        self,
        request: AgentRequest,
        response: AgentResponse,
        analysis: dict,
        protected: set[str],
        declined: set[str],
    ) -> AgentResponse:
        month = analysis.get("month")
        capacity = self._call(response, "compute_savings_capacity", {"month": month} if month else {})
        if capacity:
            response.reasoning.append(
                f"{month}: income {money(capacity['income'])}, discretionary "
                f"{money(capacity['discretionary_spend'])} "
                f"({capacity['discretionary_share_pct']}% of income), surplus "
                f"{money(capacity['surplus'])}."
            )

        categories = (analysis.get("summary") or {}).get("categories", {})
        one_offs = analysis.get("one_offs", {})
        movers = {m["category"]: m for m in (analysis.get("comparison") or {}).get("movers", [])}

        targets = self._choose_targets(categories, movers, one_offs, protected, declined, response)
        if not targets:
            response.reasoning.append(
                "No discretionary category cleared the impact threshold after constraints."
            )

        plan = None
        if targets:
            reductions = {c: cut for c, (cut, _) in targets.items()}
            plan = self._call(response, "simulate_savings_plan",
                              {"reductions": reductions, **({"month": month} if month else {})})

        if plan:
            response.reasoning.append(
                f"Simulated the plan: {money(plan['monthly_saving'])}/month, "
                f"{money(plan['annual_saving'])}/year."
            )
            lines = {line["category"]: line for line in plan["lines"]}
            for category, (cut, reason) in targets.items():
                line = lines.get(category)
                if not line or line["monthly_saving"] < MIN_MONTHLY_IMPACT:
                    continue
                response.recommendations.append(Recommendation(
                    rec_id=f"REC-{len(response.recommendations) + 1:02d}",
                    title=f"Trim {category} by {cut:.0f}%",
                    rationale=reason,
                    action=(
                        f"Cap {category} at {money(line['new_monthly'])} a month "
                        f"(down from {money(line['current_monthly'])}), saving "
                        f"{money(line['monthly_saving'])} a month."
                    ),
                    category=category,
                    monthly_impact=line["monthly_saving"],
                    effort="low" if cut <= 10 else ("medium" if cut <= 15 else "high"),
                    confidence=0.85 if category in movers else 0.7,
                    evidence={"simulation": line, "mover": movers.get(category)},
                ))

        self._subscription_recommendation(response, protected, declined)
        self._goal_recommendation(response, plan, capacity)

        response.recommendations.sort(key=lambda r: r.monthly_impact, reverse=True)
        total = sum(r.monthly_impact for r in response.recommendations)
        for rec in response.recommendations:
            self._finding(
                response,
                "recommendation",
                rec.title,
                rec.action,
                Severity.INFO,
                rec.evidence,
                ("advice", rec.category),
            )

        response.data = {
            "month": month,
            "capacity": capacity,
            "simulation": plan,
            "total_monthly_impact": round(total, 2),
            "total_annual_impact": round(total * 12, 2),
            "constraints_applied": {
                "protected": sorted(protected),
                "declined": sorted(declined),
            },
        }
        response.summary = (
            f"{len(response.recommendations)} action(s) worth {money(total)} a month "
            f"({money(total * 12)} a year)."
            if response.recommendations
            else "No material savings actions identified."
        )
        self.memory.working.put("recommendations", response.data)
        return response

    # ------------------------------------------------------------------
    def _choose_targets(
        self,
        categories: dict,
        movers: dict,
        one_offs: dict,
        protected: set[str],
        declined: set[str],
        response: AgentResponse,
    ) -> dict[str, tuple[float, str]]:
        """Pick categories to cut, sizing each cut to the evidence."""
        chosen: dict[str, tuple[float, str]] = {}
        ranked = sorted(
            (c for c in categories if c in DISCRETIONARY),
            key=lambda c: categories[c]["total"],
            reverse=True,
        )
        for category in ranked:
            if len(chosen) >= 3:
                break
            if category in protected:
                response.reasoning.append(
                    f"Skipped {category}: customer marked it protected."
                )
                continue
            if f"cut-{category.lower().replace(' & ', '-').replace(' ', '-')}" in declined:
                response.reasoning.append(
                    f"Skipped {category}: this advice was declined previously."
                )
                continue
            if category in one_offs:
                response.reasoning.append(
                    f"Skipped {category}: the increase is a one-off "
                    f"{money(one_offs[category]['largest_txn'])} purchase at "
                    f"{one_offs[category]['merchant']}, not a habit."
                )
                continue

            info = categories[category]
            mover = movers.get(category)
            # a shrinking category still qualifies for the gentlest tier
            growth = max((mover or {}).get("change_pct") or 0.0, 0.0)
            cut = next(size for floor, size in CUT_LADDER if growth >= floor)
            if info["total"] * cut / 100 < MIN_MONTHLY_IMPACT:
                continue

            if growth >= 25:
                reason = (
                    f"{category} rose {pct(growth)} month over month to "
                    f"{money(info['total'])}, driven mainly by "
                    f"{info['top_merchant']['merchant']} "
                    f"({money(info['top_merchant']['total'])}). "
                    "Bringing it back toward the previous level is the single "
                    "largest lever available."
                )
            else:
                reason = (
                    f"{category} is {info['share_pct']}% of your spend at "
                    f"{money(info['total'])} across {info['txn_count']} transactions, "
                    f"mostly {info['top_merchant']['merchant']}. A modest trim here "
                    "is low-disruption."
                )
            chosen[category] = (cut, reason)
        return chosen

    def _subscription_recommendation(
        self, response: AgentResponse, protected: set[str], declined: set[str]
    ) -> None:
        subs = self._call(response, "list_recurring_subscriptions", {"discretionary_only": True})
        if not subs or not subs["subscriptions"]:
            return
        newest = max(subs["subscriptions"], key=lambda s: s["first_seen"])
        response.reasoning.append(
            f"{subs['count']} discretionary subscription(s) costing "
            f"{money(subs['monthly_total'])}/month "
            f"({money(subs['annualised_total'])}/year)."
        )
        if "subscriptions" in declined:
            response.reasoning.append("Skipped subscription advice: previously declined.")
            return
        response.recommendations.append(Recommendation(
            rec_id=f"REC-{len(response.recommendations) + 1:02d}",
            title=f"Review {subs['count']} recurring subscriptions",
            rationale=(
                f"You have {subs['count']} active discretionary subscriptions totalling "
                f"{money(subs['monthly_total'])} a month. The newest, {newest['merchant']} "
                f"at {money(newest['monthly_amount'])}, started {newest['first_seen']}."
            ),
            action=(
                f"Cancelling the ones you no longer use would recover up to "
                f"{money(subs['monthly_total'])} a month "
                f"({money(subs['annualised_total'])} a year)."
            ),
            category="Subscriptions",
            monthly_impact=round(subs["monthly_total"], 2),
            effort="low",
            confidence=0.9,
            evidence={"subscriptions": subs["subscriptions"]},
        ))

    def _goal_recommendation(
        self, response: AgentResponse, plan: dict | None, capacity: dict | None
    ) -> None:
        extra = plan["monthly_saving"] if plan else 0.0
        base = max((capacity or {}).get("surplus", 0.0), 0.0)
        goals = self._call(response, "get_financial_goals", {"monthly_contribution": base + extra})
        if not goals or not goals["goals"]:
            return
        behind = [g for g in goals["goals"] if not g["on_track"]]
        if not behind:
            response.reasoning.append("All goals are on track at the projected contribution.")
            return
        goal = min(behind, key=lambda g: g["months_remaining"] or 999)
        response.reasoning.append(
            f"Goal '{goal['name']}' needs {money(goal['required_monthly'])}/month; "
            f"projected contribution is {money(base + extra)}."
        )
        response.recommendations.append(Recommendation(
            rec_id=f"REC-{len(response.recommendations) + 1:02d}",
            title=f"Automate a standing instruction toward '{goal['name']}'",
            rationale=(
                f"'{goal['name']}' is {goal['progress_pct']}% funded with "
                f"{goal['months_remaining']} months left, which needs "
                f"{money(goal['required_monthly'])} a month. Freeing "
                f"{money(extra)} through the cuts above closes most of that gap."
            ),
            action=(
                f"Set up an auto-debit of {money(min(goal['required_monthly'], base + extra))} "
                f"on salary day so the saving happens before it can be spent."
            ),
            category="Savings",
            monthly_impact=0.0,
            effort="low",
            confidence=0.8,
            evidence={"goal": goal},
        ))

    def _goals(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        """Funded share, required contribution and shortfall per goal."""
        month = request.params.get("month")
        capacity = self._call(response, "compute_savings_capacity",
                              {"month": month} if month else {})
        contribution = max((capacity or {}).get("surplus", 0.0), 0.0)
        invested = (capacity or {}).get("invested", 0.0)
        response.reasoning.append(
            f"Assuming {money(contribution)}/month of free surplus plus "
            f"{money(invested)} already invested each month."
        )

        goals = self._call(response, "get_financial_goals",
                           {"monthly_contribution": contribution + invested})
        if not goals or not goals["goals"]:
            response.status = "partial"
            response.summary = "No savings goals are registered on this account."
            return response

        behind = []
        for goal in goals["goals"]:
            shortfall = round(goal["required_monthly"] - (contribution + invested), 2)
            on_track = goal["on_track"]
            if not on_track:
                behind.append(goal)
            self._finding(
                response,
                "goal_progress",
                f"{goal['name']}: {goal['progress_pct']}% funded"
                + ("" if on_track else " - behind schedule"),
                (
                    f"{money(goal['saved_amount'])} of {money(goal['target_amount'])} saved, "
                    f"{money(goal['gap'])} to go by {goal['target_date']} "
                    f"({goal['months_remaining']} months). Needs "
                    f"{money(goal['required_monthly'])} a month"
                    + (
                        f"; you are short by {money(shortfall)} at the current rate."
                        if shortfall > 0
                        else " and you are contributing enough."
                    )
                ),
                Severity.HIGH if not on_track and goal.get("months_remaining", 99) <= 12
                else (Severity.MEDIUM if not on_track else Severity.INFO),
                {**goal, "shortfall_monthly": shortfall,
                 "assumed_contribution": round(contribution + invested, 2)},
                ("goal",),
            )

        if behind:
            goal = min(behind, key=lambda g: g["months_remaining"] or 999)
            response.recommendations.append(Recommendation(
                rec_id=f"REC-G{len(response.recommendations) + 1:02d}",
                title=f"Close the gap on '{goal['name']}'",
                rationale=(
                    f"'{goal['name']}' needs {money(goal['required_monthly'])} a month "
                    f"to land by {goal['target_date']}, against the "
                    f"{money(contribution + invested)} you are putting aside now."
                ),
                action=(
                    f"Either raise the monthly contribution to "
                    f"{money(goal['required_monthly'])}, or move the target date out. "
                    "Freeing money from discretionary spending is the usual first step."
                ),
                category="Savings",
                monthly_impact=0.0,
                effort="medium",
                confidence=0.85,
                evidence={"goal": goal},
            ))

        response.data = {
            "assumed_monthly_contribution": round(contribution + invested, 2),
            "capacity": capacity,
            "goals": goals["goals"],
            "behind_count": len(behind),
        }
        response.summary = (
            f"{len(goals['goals'])} goal(s) reviewed; {len(behind)} behind schedule."
        )
        self.memory.working.put("goals", response.data)
        return response

    def _subscriptions(
        self, response: AgentResponse, protected: set[str], declined: set[str]
    ) -> AgentResponse:
        self._subscription_recommendation(response, protected, declined)
        subs = self.memory.working.get("subscription.audit")
        total = sum(r.monthly_impact for r in response.recommendations)
        for rec in response.recommendations:
            self._finding(response, "recommendation", rec.title, rec.action,
                          Severity.INFO, rec.evidence, ("advice", "subscriptions"))
        response.data = {"total_monthly_impact": round(total, 2)}
        response.summary = (
            f"Subscription audit: {money(total)} a month is recurring and reviewable."
            if total else "No discretionary subscriptions found."
        )
        self.memory.working.put("recommendations", response.data)
        return response
