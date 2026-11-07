"""HTTP layer over `AdvisorRuntime`, plus the single-page UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import analytics as an
from .bank import CustomerNotFound
from .categorizer import categorize_all, detect_recurring
from .config import settings
from .orchestrator import AdvisorRuntime

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="Personal Finance Advisor Agent",
    description=(
        "Agentic AI prototype for retail banking: a planner agent orchestrates "
        "spending-analysis, alerting and recommendation agents over mock "
        "core-banking data."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = AdvisorRuntime()


class ChatRequest(BaseModel):
    customer_id: str = Field(default="CUST001")
    message: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default="web")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "as_of": runtime.bank.as_of.isoformat(),
        "llm_enabled": settings.llm_enabled,
        "model": settings.model if settings.llm_enabled else None,
        "customers": len(runtime.customers()),
    }


@app.get("/api/customers")
def customers() -> dict[str, Any]:
    rows = []
    for customer in runtime.customers():
        profile = runtime.bank.get_profile(customer["customer_id"])
        rows.append({
            **customer,
            "monthly_income": profile["monthly_income"],
            "balance": profile["accounts"][0]["balance"],
        })
    return {"customers": rows, "as_of": runtime.bank.as_of.isoformat()}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return runtime.capabilities()


@app.get("/api/dashboard/{customer_id}")
def dashboard(customer_id: str) -> dict[str, Any]:
    """Everything the UI needs for the account overview, in one call."""
    try:
        profile = runtime.bank.get_profile(customer_id)
        budgets = runtime.bank.get_budgets(customer_id)
        goals = runtime.bank.get_goals(customer_id)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    as_of = runtime.bank.as_of
    txns = categorize_all(runtime.bank.get_transactions(customer_id))
    current = an.month_key(as_of)
    previous = an.shift_month(current, -1)

    month_rows = [t for t in txns if t.month == previous]
    return {
        "profile": {
            "customer_id": profile["customer_id"],
            "name": profile["name"],
            "segment": profile["segment"],
            "city": profile["city"],
            "monthly_income": profile["monthly_income"],
            "balance": profile["accounts"][0]["balance"],
        },
        "as_of": as_of.isoformat(),
        "current_month": current,
        "reference_month": previous,
        "monthly_totals": an.by_month(txns),
        "categories": an.by_category(month_rows),
        "total_spend": an.total_spend(month_rows),
        "comparison": an.compare_months(txns, previous, an.shift_month(previous, -1)),
        "budgets": an.budget_status(txns, budgets, current, as_of),
        "capacity": an.savings_capacity(txns, previous, profile["monthly_income"]),
        "goals": [
            an.goal_projection(
                g,
                max(an.savings_capacity(txns, previous, profile["monthly_income"])["surplus"], 0),
                as_of,
            )
            for g in goals
        ],
        "subscriptions": detect_recurring(txns),
        "transaction_count": len(txns),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        turn = runtime.ask(request.customer_id, request.message, request.session_id)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return turn.to_dict()


@app.post("/api/reset")
def reset(request: ChatRequest) -> dict[str, Any]:
    """Drop the conversation; long-term memory is kept."""
    runtime.reset(request.customer_id, request.session_id)
    return {"reset": True, "customer_id": request.customer_id}


@app.post("/api/memory/clear")
def clear_memory(request: ChatRequest) -> dict[str, Any]:
    """Wipe long-term memory too, so the demo can be replayed."""
    session = runtime.session(request.customer_id, request.session_id)
    for note in list(session.memory.profile.all()):
        session.memory.profile.forget(note.kind, note.key)
    runtime.reset(request.customer_id, request.session_id)
    return {"cleared": True, "customer_id": request.customer_id}


@app.get("/api/suggestions")
def suggestions() -> dict[str, Any]:
    return {
        "suggestions": [
            "How can I improve my monthly savings?",
            "Where did my money go last month?",
            "Am I over budget anywhere?",
            "Has anything unusual happened on my account?",
            "What subscriptions am I paying for?",
            "Am I on track for my goals?",
            "Don't suggest cutting my travel spending",
        ]
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/favicon.ico")
def favicon() -> JSONResponse:
    return JSONResponse({}, status_code=204)
