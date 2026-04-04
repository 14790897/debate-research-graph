from __future__ import annotations

import asyncio
import argparse
import json
import os
from datetime import datetime
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
    parser.add_argument("--topic", help="Debate topic.")
    parser.add_argument("--turns", type=int, default=2, help="Maximum debate rounds.")
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
        "--search-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fetch full text for each search result and extract summaries.",
    )
    parser.add_argument(
        "--search-max-chars",
        type=int,
        default=1200,
        help="Max characters per fetched document when --search-full is enabled.",
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
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each turn as it is generated.",
    )
    parser.add_argument(
        "--diagram",
        default="",
        help="Write a Mermaid flowchart to the given file path and exit.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write debate transcript and final report to this file.",
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


def _format_api_error(exc: APIError) -> str:
    lines = [
        "Model request failed.",
        f"- error type: {exc.__class__.__name__}",
        f"- message: {exc}",
    ]

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        lines.append(f"- status code: {status_code}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        lines.append(f"- request id: {request_id}")

    if exc.code:
        lines.append(f"- code: {exc.code}")
    if exc.type:
        lines.append(f"- type: {exc.type}")
    if exc.param:
        lines.append(f"- param: {exc.param}")

    request = getattr(exc, "request", None)
    if request is not None:
        lines.append(f"- request method: {request.method}")
        lines.append(f"- request url: {request.url}")

    body = getattr(exc, "body", None)
    if body is not None:
        if isinstance(body, (dict, list)):
            rendered_body = json.dumps(body, ensure_ascii=False, indent=2)
        else:
            rendered_body = str(body)
        lines.append("- response body:")
        lines.append(rendered_body)

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        lines.append(f"- cause type: {cause.__class__.__name__}")
        lines.append(f"- cause: {cause}")

    notes = getattr(exc, "__notes__", None)
    if notes:
        lines.append("- notes:")
        lines.extend(str(note) for note in notes)

    return "\n".join(lines)


def _default_output_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", f"debate_{timestamp}.md")


def _open_output_file(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "w", encoding="utf-8")


async def amain() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.diagram and not args.topic:
        parser.error("--topic is required unless --diagram is provided.")

    if args.turns < 1:
        parser.error("--turns must be at least 1")

    if args.max_retries < 0:
        parser.error("--max-retries must be at least 0")

    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    if args.diagram:
        class _NoopModel:
            async def ainvoke(self, messages: list[object]) -> object:
                raise RuntimeError("This model should not be invoked for diagram output.")

        graph = build_debate_graph(
            model=_NoopModel(),
            researcher=WebResearcher(SearchConfig(enabled=False)),
        )
        diagram = graph.get_graph().draw_mermaid()
        with open(args.diagram, "w", encoding="utf-8") as handle:
            handle.write(diagram)
        print(f"Flowchart written to {args.diagram}")
        return 0

    if not args.api_key:
        parser.error(
            "Missing API key. Set OPENAI_API_KEY in environment variables or pass --api-key."
        )

    output_path = args.output or _default_output_path()
    output_handle = _open_output_file(output_path)

    def _write_line(text: str) -> None:
        output_handle.write(text + "\n")
        output_handle.flush()

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
                fetch_full_text=args.search_full,
                max_chars_per_doc=args.search_max_chars,
            )
        )
        def on_update(turn: dict[str, object]) -> None:
            role = turn["role"]
            round_label = "开场" if turn["round"] == 0 else f"第 {turn['round']} 轮"
            print(f"\n[{round_label}][{role}]")
            print(turn["content"])
            print()
            _write_line(f"## [{round_label}][{role}]")
            _write_line(str(turn["content"]))
            _write_line("")

        graph = build_debate_graph(
            model=model,
            researcher=researcher,
            on_update=on_update if args.live else None,
        )

        _write_line(f"# 议题\n{args.topic}\n")
        _write_line(
            f"# 参数\nturns={args.turns}, search={args.search}, model={args.model}\n"
        )

        initial_state: DebateState = {
            "topic": args.topic,
            "dialogue_history": [],
            "current_turn": 0,
            "max_turns": args.turns,
            "search_context": "",
            "final_report": "",
        }
        final_state = await graph.ainvoke(initial_state)

        _write_line("# 完整记录")
        for turn in final_state["dialogue_history"]:
            round_label = "开场" if turn["round"] == 0 else f"第 {turn['round']} 轮"
            _write_line(f"## [{round_label}][{turn['role']}]")
            _write_line(str(turn["content"]))
            _write_line("")

        _write_line("# 最终报告")
        _write_line(final_state["final_report"] or "未生成最终报告。")
    except APIError as exc:
        print(
            f"{_format_api_error(exc)}\n"
            f"- configured base url: {args.base_url or 'default OpenAI endpoint'}\n"
            f"- configured model: {args.model}\n"
            "This is usually an upstream service issue, a custom endpoint compatibility issue, "
            "or an unsupported model name on that endpoint.",
            file=sys.stderr,
        )
        return 1
    finally:
        output_handle.close()

    print(f"Saved transcript to {output_path}")
    _print_result(final_state)
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
