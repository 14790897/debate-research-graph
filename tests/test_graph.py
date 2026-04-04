from __future__ import annotations

import asyncio
import unittest

from deep_research.graph import build_debate_graph
from deep_research.search import WebResearcher
from deep_research.state import DebateState


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> FakeResponse:
        self.calls.append(messages)
        if not self.outputs:
            raise AssertionError("FakeModel ran out of scripted outputs.")
        return FakeResponse(self.outputs.pop(0))

    def invoke(self, messages: list[object]) -> FakeResponse:
        return asyncio.run(self.ainvoke(messages))


class FakeResearcher(WebResearcher):
    def __init__(self, content: str) -> None:
        self.content = content

    def build_context(self, topic: str) -> str:
        return self.content

    async def build_context_async(self, topic: str) -> str:
        return self.content


class DebateGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_runs_until_final_round(self) -> None:
        model = FakeModel(
            [
                "正方第1轮",
                "反方第1轮",
                "裁判第1轮",
                "正方第2轮",
                "反方第2轮",
                "# 综合结论\n最终结论",
            ]
        )
        researcher = FakeResearcher("外部资料")
        graph = build_debate_graph(model=model, researcher=researcher)

        initial_state: DebateState = {
            "topic": "AI 是否会彻底取代初级程序员？",
            "dialogue_history": [],
            "current_turn": 0,
            "max_turns": 2,
            "search_context": "",
            "final_report": "",
        }

        result = await graph.ainvoke(initial_state)

        self.assertEqual(result["current_turn"], 2)
        self.assertEqual(result["search_context"], "外部资料")
        self.assertEqual(result["final_report"], "# 综合结论\n最终结论")
        self.assertEqual(len(result["dialogue_history"]), 7)
        self.assertEqual(result["dialogue_history"][0]["role"], "moderator")
        self.assertEqual(result["dialogue_history"][-1]["role"], "moderator")


if __name__ == "__main__":
    unittest.main()
