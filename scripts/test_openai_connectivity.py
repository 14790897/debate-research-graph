from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import APIError, OpenAI
from dotenv import load_dotenv


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _print_block(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def _format_api_error(exc: APIError) -> str:
    lines = [
        f"error type: {exc.__class__.__name__}",
        f"message: {exc}",
    ]

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        lines.append(f"status code: {status_code}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        lines.append(f"request id: {request_id}")

    if exc.code:
        lines.append(f"code: {exc.code}")
    if exc.type:
        lines.append(f"type: {exc.type}")
    if exc.param:
        lines.append(f"param: {exc.param}")

    request = getattr(exc, "request", None)
    if request is not None:
        lines.append(f"request method: {request.method}")
        lines.append(f"request url: {request.url}")

    body = getattr(exc, "body", None)
    if body is not None:
        lines.append("response body:")
        if isinstance(body, (dict, list)):
            lines.append(json.dumps(body, ensure_ascii=False, indent=2))
        else:
            lines.append(str(body))

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        lines.append(f"cause type: {cause.__class__.__name__}")
        lines.append(f"cause: {cause}")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Test connectivity against an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--base-url",
        default=_first_env("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_URL"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=_first_env("OPENAI_API_KEY"),
        help="API key.",
    )
    parser.add_argument(
        "--model",
        default=_first_env("OPENAI_MODEL", "DEBATE_MODEL") or "gpt-4.1",
        help="Model name to test with chat completions.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only test models listing, skip chat completions.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: pong",
        help="Prompt used for chat completion test.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=0,
        timeout=args.timeout,
    )

    print("OpenAI connectivity test")
    print(f"base url: {args.base_url or 'default OpenAI endpoint'}")
    print(f"model: {args.model}")
    print(f"timeout: {args.timeout}s")

    try:
        models = client.models.list()
        model_ids = [item.id for item in models.data[:10]]
        _print_block(
            "GET /models OK",
            {
                "count": len(models.data),
                "first_models": model_ids,
            },
        )
    except APIError as exc:
        _print_block("GET /models FAILED", _format_api_error(exc))
        return 1

    if args.skip_chat:
        return 0

    try:
        completion = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
        )
        message = completion.choices[0].message.content
        _print_block(
            "POST /chat/completions OK",
            {
                "id": completion.id,
                "model": completion.model,
                "message": message,
            },
        )
        return 0
    except APIError as exc:
        _print_block("POST /chat/completions FAILED", _format_api_error(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
