from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ddgs import DDGS
import trafilatura


@dataclass(slots=True)
class SearchConfig:
    enabled: bool = False
    max_results: int = 5
    region: str = "us-en"
    timeout: int = 15
    fetch_full_text: bool = False
    max_chars_per_doc: int = 1200


class WebResearcher:
    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()

    def _fetch_full_text(self, url: str) -> str:
        downloaded = trafilatura.fetch_url(url, timeout=self.config.timeout)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not extracted:
            return ""
        extracted = extracted.strip()
        if len(extracted) > self.config.max_chars_per_doc:
            return extracted[: self.config.max_chars_per_doc] + "..."
        return extracted

    def build_context(self, topic: str) -> str:
        if not self.config.enabled:
            return ""

        query = f"{topic} 研究 数据 统计 证据 争议"

        try:
            results = DDGS(timeout=self.config.timeout).text(
                query,
                region=self.config.region,
                safesearch="moderate",
                max_results=self.config.max_results,
            )
        except Exception as exc:
            return f"外部检索失败，本轮按无检索上下文处理。错误信息：{exc}"

        if not results:
            return "没有检索到可用外部资料。"

        blocks: list[str] = []
        for index, item in enumerate(results, start=1):
            title = item.get("title") or "Untitled"
            url = item.get("href") or item.get("url") or ""
            snippet = item.get("body") or "无摘要"
            full_text = ""
            if self.config.fetch_full_text and url:
                try:
                    full_text = self._fetch_full_text(url)
                except Exception:
                    full_text = ""
            blocks.append(
                f"{index}. {title}\n"
                f"URL: {url}\n"
                f"摘要: {snippet}"
            )
            if full_text:
                blocks.append(f"原文摘要: {full_text}")

        return "\n\n".join(blocks)

    async def build_context_async(self, topic: str) -> str:
        return await asyncio.to_thread(self.build_context, topic)
