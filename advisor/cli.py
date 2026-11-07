"""Command-line interface.

    python -m advisor.cli chat                 interactive session
    python -m advisor.cli ask "<question>"     one question, full trace
    python -m advisor.cli demo                 scripted multi-turn walkthrough
    python -m advisor.cli agents               capability registry
    python -m advisor.cli customers            available mock customers
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import settings
from .orchestrator import AdvisorRuntime, AdvisorTurn

BOLD, DIM, CYAN, YELLOW, GREEN, RED, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m"
)

SEVERITY_COLOUR = {
    "critical": RED, "high": RED, "medium": YELLOW, "low": CYAN, "info": DIM,
}

DEMO_SCRIPT = [
    ("A broad question - the planner picks two agents and chains them",
     "How can I improve my monthly savings?"),
    ("A follow-up with no period stated - resolved from the conversation",
     "What about this month?"),
    ("Proactive monitoring - this one triggers a re-plan",
     "Am I over budget anywhere?"),
    ("The customer pushes back; this is written to long-term memory",
     "Don't suggest cutting my travel spending"),
    ("Same question as before - the advice now respects that preference",
     "How can I improve my monthly savings?"),
    ("Anomaly detection over the raw transaction book",
     "Has anything unusual happened on my account?"),
    ("Out of scope - the advisor declines without inventing an answer",
     "What is the weather in Mumbai?"),
]


def banner(text: str) -> None:
    print(f"\n{BOLD}{'=' * 78}\n{text}\n{'=' * 78}{RESET}")


def print_turn(turn: AdvisorTurn, show_trace: bool = True) -> None:
    if show_trace:
        print(f"\n{DIM}--- reasoning trace ---{RESET}")
        for event in turn.trace:
            print(
                f"{DIM}[{event['elapsed_ms']:>5}ms]{RESET} "
                f"{CYAN}{event['stage'].upper():<11}{RESET}{event['label']}"
            )
            if event["detail"]:
                for line in event["detail"].split("\n"):
                    print(f"{DIM}{'':>9}   {line}{RESET}")

    print(f"\n{BOLD}{GREEN}Advisor:{RESET}\n{turn.answer}\n")

    if turn.alerts:
        print(f"{BOLD}Alerts{RESET}")
        for alert in turn.alerts:
            colour = SEVERITY_COLOUR.get(alert["severity"], "")
            print(f"  {colour}[{alert['severity']:<8}]{RESET} {alert['title']}")

    if turn.recommendations:
        print(f"{BOLD}Recommendations{RESET}")
        for rec in turn.recommendations:
            impact = f"+₹{rec['monthly_impact']:,.0f}/mo" if rec["monthly_impact"] else "enabler"
            print(f"  - {rec['title']:<46} {impact:>14}  [{rec['effort']} effort]")

    steps = turn.plan.get("steps", [])
    print(
        f"\n{DIM}intent={turn.intent['intent']} ({turn.intent['confidence']} via "
        f"{turn.intent['method']}) | steps={len(steps)} | "
        f"tools={len(turn.tool_calls)} | llm_calls={turn.llm['calls']} | "
        f"{turn.duration_ms}ms{RESET}"
    )


def cmd_chat(runtime: AdvisorRuntime, args: argparse.Namespace) -> int:
    session = runtime.session(args.customer, args.session)
    profile = runtime.bank.get_profile(args.customer)
    banner(f"Personal Finance Advisor - {profile['name']} ({args.customer})")
    print(
        f"Data as of {runtime.bank.as_of}. "
        f"LLM: {'on - ' + settings.model if settings.llm_enabled else 'off (deterministic)'}\n"
        f"{DIM}Type a question, 'trace' to toggle the trace, or 'exit'.{RESET}"
    )
    show_trace = args.trace
    while True:
        try:
            message = input(f"\n{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not message:
            continue
        if message.lower() in {"exit", "quit", ":q"}:
            return 0
        if message.lower() == "trace":
            show_trace = not show_trace
            print(f"{DIM}trace {'on' if show_trace else 'off'}{RESET}")
            continue
        print_turn(session.ask(message), show_trace)


def cmd_ask(runtime: AdvisorRuntime, args: argparse.Namespace) -> int:
    turn = runtime.ask(args.customer, args.question, args.session)
    if args.json:
        print(json.dumps(turn.to_dict(), indent=2, default=str))
        return 0
    banner(f"Q: {args.question}")
    print_turn(turn, show_trace=not args.quiet)
    return 0


def cmd_demo(runtime: AdvisorRuntime, args: argparse.Namespace) -> int:
    session = runtime.session(args.customer, "demo")
    profile = runtime.bank.get_profile(args.customer)
    banner(
        f"Scripted walkthrough - {profile['name']} ({args.customer})\n"
        f"Data as of {runtime.bank.as_of} | "
        f"LLM {'on' if settings.llm_enabled else 'off (deterministic)'}"
    )
    for index, (note, question) in enumerate(DEMO_SCRIPT, start=1):
        banner(f"[{index}/{len(DEMO_SCRIPT)}] {note}\nQ: {question}")
        print_turn(session.ask(question), show_trace=not args.quiet)
    banner("Walkthrough complete")
    return 0


def cmd_agents(runtime: AdvisorRuntime, args: argparse.Namespace) -> int:
    info = runtime.capabilities()
    banner("Agent registry")
    for agent in info["agents"]:
        print(f"{BOLD}{agent['name']}{RESET}\n  {agent['description']}")
        print(f"  {DIM}capabilities:{RESET} {', '.join(agent['capabilities']) or '-'}")
        print(f"  {DIM}tasks:{RESET}        {', '.join(agent['tasks']) or '-'}\n")
    banner(f"Tools ({len(info['tools'])})")
    for tool in info["tools"]:
        print(f"  {tool['name']:<30} {tool['description']}")
    banner(f"Intents ({len(info['intents'])})")
    for intent in info["intents"]:
        print(f"  {intent['intent']:<22} requires: {', '.join(intent['requires']) or '-'}")
    return 0


def cmd_customers(runtime: AdvisorRuntime, args: argparse.Namespace) -> int:
    banner("Mock customers")
    for customer in runtime.customers():
        profile = runtime.bank.get_profile(customer["customer_id"])
        print(
            f"  {customer['customer_id']}  {customer['name']:<16} "
            f"{customer['segment']:<22} {customer['city']:<12} "
            f"income ₹{profile['monthly_income']:,.0f}/mo"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advisor",
        description="Agentic AI Personal Finance Advisor (retail banking prototype)",
    )
    parser.add_argument("--customer", default="CUST001", help="customer id (default CUST001)")
    parser.add_argument("--session", default="cli", help="session id for memory scoping")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive conversation")
    chat.add_argument("--trace", action="store_true", default=True)
    chat.set_defaults(func=cmd_chat)

    ask = sub.add_parser("ask", help="ask a single question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="emit the full turn as JSON")
    ask.add_argument("--quiet", action="store_true", help="hide the reasoning trace")
    ask.set_defaults(func=cmd_ask)

    demo = sub.add_parser("demo", help="scripted multi-turn walkthrough")
    demo.add_argument("--quiet", action="store_true", help="hide the reasoning trace")
    demo.set_defaults(func=cmd_demo)

    sub.add_parser("agents", help="show agents, tools and intents").set_defaults(func=cmd_agents)
    sub.add_parser("customers", help="list mock customers").set_defaults(func=cmd_customers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = AdvisorRuntime()
    return args.func(runtime, args)


if __name__ == "__main__":
    sys.exit(main())
