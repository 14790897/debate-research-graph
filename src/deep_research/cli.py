from __future__ import annotations

import asyncio
import argparse
import os
import sys

from langchain_openai import ChatOpenAI
from openai import APIError

from deep_research.graph import build_debate_graph
from deep_research.search import SearchConfig, WebResearcher
from deep_research.state import DebateState


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a LangGraph multi-agent debate for deep research."
    )
    parser.add_argument("--topic", required=True, help="Debate topic.")
    parser.add_argument("--turns", type=int, default=3, help="Maximum debate rounds.")
    parser.add_argument(
        "--model",
        default=_first_env("DEBATE_MODEL", "OPENAI_MODEL") or "gpt-4.1",
        help="Chat model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable web research before the debate starts.",
    )
    parser.add_argument(
        "--search-results",
        type=int,
        default=5,
        help="Number of web results to load into the debate context.",
    )
    parser.add_argument(
        "--base-url",
        default=_first_env("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_URL"),
        help="Optional OpenAI-compatible base URL. Defaults to environment variables.",
    )
    parser.add_argument(
        "--api-key",
        default=_first_env("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to environment variables.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=_env_int("OPENAI_MAX_RETRIES", 5),
        help="Maximum OpenAI client retries for transient errors.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_float("OPENAI_TIMEOUT", 60.0),
        help="Request timeout in seconds.",
    )
    return parser


def _format_role(role: str) -> str:
    mapping = {
        "moderator": "裁判",
        "proponent": "正方",
        "opponent": "反方",
    }
    return mapping.get(role, role)


def _print_result(state: DebateState) -> None:
    print("\n=== 辩论记录 ===\n")
    for turn in state["dialogue_history"]:
        round_label = "开场" if turn["round"] == 0 else f"第 {turn['round']} 轮"
        print(f"[{round_label}][{_format_role(turn['role'])}]")
        print(turn["content"])
        print()

    print("=== 最终报告 ===\n")
    print(state["final_report"] or "未生成最终报告。")


async def amain() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.turns < 1:
        parser.error("--turns must be at least 1")

    if args.max_retries < 0:
        parser.error("--max-retries must be at least 0")

    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    if not args.api_key:
        parser.error(
            "Missing API key. Set OPENAI_API_KEY in environment variables or pass --api-key."
        )

    try:
        model = ChatOpenAI(
            model=args.model,
            temperature=args.temperature,
            api_key=args.api_key,
            base_url=args.base_url,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        researcher = WebResearcher(
            SearchConfig(
                enabled=args.search,
                max_results=args.search_results,
            )
        )
        graph = build_debate_graph(model=model, researcher=researcher)

        initial_state: DebateState = {
            "topic": args.topic,
            "dialogue_history": [],
            "current_turn": 0,
            "max_turns": args.turns,
            "search_context": "",
            "final_report": "",
        }
        final_state = await graph.ainvoke(initial_state)
    except APIError as exc:
        print(
            "Model request failed.\n"
            f"- error type: {exc.__class__.__name__}\n"
            f"- message: {exc}\n"
            f"- base url: {args.base_url or 'default OpenAI endpoint'}\n"
            f"- model: {args.model}\n"
            "This is usually an upstream service issue, a custom endpoint compatibility issue, "
            "or an unsupported model name on that endpoint.",
            file=sys.stderr,
        )
        return 1

    _print_result(final_state)
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
