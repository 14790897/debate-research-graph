from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ddgs import DDGS
import trafilatura
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(slots=True)
class SearchConfig:
    enabled: bool = False
    max_results: int = 5
    region: str = "us-en"
    timeout: int = 15
    fetch_full_text: bool = False
    max_chars_per_doc: int = 1200
    min_title_chars: int = 5
    min_snippet_chars: int = 30
    ai_filter: bool = False


class WebResearcher:
    def __init__(self, config: SearchConfig | None = None, model: Any | None = None) -> None:
        self.config = config or SearchConfig()
        self.model = model

    def _is_low_quality(self, title: str, url: str, snippet: str) -> bool:
        if not title or not url:
            return True
        if not (url.startswith("http://") or url.startswith("https://")):
            return True
        if len(title.strip()) < self.config.min_title_chars:
            return True
        if len(snippet.strip()) < self.config.min_snippet_chars:
            return True
        return False

    def _normalize_results(self, results: list[dict[str, Any]]) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in results:
            title = item.get("title") or "Untitled"
            url = item.get("href") or item.get("url") or ""
            snippet = item.get("body") or "无摘要"
            key = f"{title.strip().lower()}|{url.strip().lower()}"
            if self._is_low_quality(title, url, snippet) or key in seen:
                continue
            seen.add(key)
            blocks.append({"title": title, "url": url, "snippet": snippet})
        return blocks

    def _coerce_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    async def _ai_keep_result(self, title: str, url: str, snippet: str) -> bool:
        if not self.model:
            return True
        system_prompt = (
            "你是检索结果质量审核员，任务是过滤低质量来源。"
            "低质量包括：标题党、营销软文、无来源、无实质内容、"
            "明显抄袭拼接、论坛/贴吧式碎片发言、内容与标题不符。"
            "只输出 KEEP 或 DROP，不要输出其他文字。"
        )
        user_prompt = (
            f"标题: {title}\n"
            f"URL: {url}\n"
            f"摘要: {snippet}\n"
        )
        response = await self.model.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        text = self._coerce_text(getattr(response, "content", response)).upper()
        if "KEEP" in text:
            return True
        if "DROP" in text:
            return False
        return False

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

        cleaned = self._normalize_results(list(results))
        if not cleaned:
            return "检索结果被低质量过滤规则全部剔除。"

        blocks: list[str] = []
        for index, item in enumerate(cleaned, start=1):
            title = item["title"]
            url = item["url"]
            snippet = item["snippet"]
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
        if not self.config.ai_filter or not self.model:
            return await asyncio.to_thread(self.build_context, topic)

        if not self.config.enabled:
            return ""

        query = f"{topic} 研究 数据 统计 证据 争议"

        try:
            results = await asyncio.to_thread(
                lambda: list(
                    DDGS(timeout=self.config.timeout).text(
                        query,
                        region=self.config.region,
                        safesearch="moderate",
                        max_results=self.config.max_results,
                    )
                )
            )
        except Exception as exc:
            return f"外部检索失败，本轮按无检索上下文处理。错误信息：{exc}"

        if not results:
            return "没有检索到可用外部资料。"

        cleaned = self._normalize_results(results)
        if not cleaned:
            return "检索结果被低质量过滤规则全部剔除。"

        kept: list[dict[str, str]] = []
        for item in cleaned:
            try:
                keep = await self._ai_keep_result(item["title"], item["url"], item["snippet"])
            except Exception:
                keep = False
            if keep:
                kept.append(item)

        if not kept:
            return "AI 审核认为检索结果质量不足，已全部过滤。"

        blocks: list[str] = []
        for index, item in enumerate(kept, start=1):
            title = item["title"]
            url = item["url"]
            snippet = item["snippet"]
            full_text = ""
            if self.config.fetch_full_text and url:
                try:
                    full_text = await asyncio.to_thread(self._fetch_full_text, url)
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
