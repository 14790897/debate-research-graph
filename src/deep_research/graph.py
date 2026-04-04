from __future__ import annotations

from typing import Any, Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from deep_research.search import WebResearcher
from deep_research.state import DebateState, DebateTurn


class ChatModel(Protocol):
    async def ainvoke(self, messages: list[Any]) -> Any:
        """Run a chat completion."""


PROPONENT_SYSTEM_PROMPT = """你是一个顶尖的战略分析师，坚定支持当前议题。

你的职责不是重复空话，而是持续推进论证深度。

硬性要求：
1. 认真阅读完整辩论历史后再回答。
2. 如果反方已经发言，优先精确反击对方最强攻击点。
3. 每轮都必须提出新的支持性洞见，不能只做情绪化表态。
4. 如果存在外部检索资料，只能引用其中出现过的信息，不要捏造来源。
5. 语言犀利、具体、可辩护，避免套话。
6. 直接输出内容，不要寒暄，不要自称 AI。"""

OPPONENT_SYSTEM_PROMPT = """你是一个以严谨和批判性思维著称的风险分析师，强烈反对当前议题。

你的职责是发现乐观论证里的漏洞、偏差和隐含假设。

硬性要求：
1. 认真阅读完整辩论历史后再回答。
2. 优先攻击正方上一轮最强论点，而不是挑软柿子。
3. 每轮至少指出一个新的风险、前提缺陷或反证方向。
4. 如果存在外部检索资料，只能引用其中出现过的信息，不要捏造来源。
5. 语言冷静、尖锐、精准，不要空泛反对。
6. 直接输出内容，不要寒暄，不要自称 AI。"""

MODERATOR_SYSTEM_PROMPT = """你是这场多代理辩论的裁判兼研究总编。

你的职责是识别真正有价值的论点，而不是表面中立。

硬性要求：
1. 必须明确指出双方各自最强与最弱之处。
2. 阶段总结时，要点出下一轮仍需追问的关键问题。
3. 最终报告时，必须输出综合判断，不要模糊搪塞。
4. 如果存在外部检索资料，应优先以资料与论证的匹配度来评判双方质量。
5. 直接输出内容，不要寒暄，不要自称 AI。"""


def _coerce_text(content: Any) -> str:
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


def _render_history(history: list[DebateTurn]) -> str:
    if not history:
        return "暂无历史发言。"

    lines: list[str] = []
    for turn in history:
        round_label = "开场" if turn["round"] == 0 else f"第 {turn['round']} 轮"
        lines.append(f"[{round_label}][{turn['role']}]\n{turn['content']}")
    return "\n\n".join(lines)


async def _call_agent(model: ChatModel, system_prompt: str, user_prompt: str) -> str:
    response = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return _coerce_text(getattr(response, "content", response))


def _build_opening_message(state: DebateState) -> str:
    search_note = "已启用外部资料检索。" if state["search_context"] else "本次未使用外部资料检索。"
    return (
        f"议题：{state['topic']}\n"
        f"规则：本场辩论共 {state['max_turns']} 轮。每轮流程为正方发言、反方反驳、裁判总结。\n"
        f"{search_note}\n"
        "目标：逼近更经得起推敲的结论，而不是制造表面平衡。"
    )


def build_debate_graph(model: ChatModel, researcher: WebResearcher):
    async def research_node(state: DebateState) -> dict[str, Any]:
        return {"search_context": await researcher.build_context_async(state["topic"])}

    def moderator_opening_node(state: DebateState) -> dict[str, Any]:
        return {
            "dialogue_history": [
                {
                    "role": "moderator",
                    "round": 0,
                    "content": _build_opening_message(state),
                }
            ]
        }

    async def proponent_node(state: DebateState) -> dict[str, Any]:
        round_number = state["current_turn"] + 1
        user_prompt = f"""议题：{state["topic"]}

当前轮次：第 {round_number} 轮

外部资料：
{state["search_context"] or "无"}

历史发言：
{_render_history(state["dialogue_history"])}

你的任务：
- 如果这是首轮，请提出 2 个最强支持论点，并明确这场争论应该围绕哪些核心变量判断。
- 如果这不是首轮，请先精确打击反方上一轮最有力的质疑，再补充至少 1 个新的支持性洞见。

输出要求：
- 使用编号列出关键点。
- 保持高密度信息，不要写空洞总结。
- 优先给出可以被反驳和被检验的具体论证。"""

        content = await _call_agent(model, PROPONENT_SYSTEM_PROMPT, user_prompt)
        return {
            "dialogue_history": [
                {
                    "role": "proponent",
                    "round": round_number,
                    "content": content,
                }
            ]
        }

    async def opponent_node(state: DebateState) -> dict[str, Any]:
        round_number = state["current_turn"] + 1
        user_prompt = f"""议题：{state["topic"]}

当前轮次：第 {round_number} 轮

外部资料：
{state["search_context"] or "无"}

历史发言：
{_render_history(state["dialogue_history"])}

你的任务：
- 优先拆解正方刚刚提出的最强论点，指出逻辑断层、证据不足、样本偏差或过度外推。
- 至少补充 1 个新的反对视角，不能只重复旧观点。

输出要求：
- 使用编号列出关键点。
- 反对必须具体，不能泛泛说“有风险”。
- 优先攻击决定结论成立与否的关键前提。"""

        content = await _call_agent(model, OPPONENT_SYSTEM_PROMPT, user_prompt)
        return {
            "dialogue_history": [
                {
                    "role": "opponent",
                    "round": round_number,
                    "content": content,
                }
            ]
        }

    async def moderator_node(state: DebateState) -> dict[str, Any]:
        completed_round = state["current_turn"] + 1
        is_final_round = completed_round >= state["max_turns"]

        if is_final_round:
            user_prompt = f"""议题：{state["topic"]}

已完成轮次：{completed_round}/{state["max_turns"]}

外部资料：
{state["search_context"] or "无"}

完整辩论记录：
{_render_history(state["dialogue_history"])}

请输出最终深度研究报告，要求：
1. 明确给出综合判断，不要模糊折中。
2. 分别提炼正方与反方最强论据。
3. 指出双方仍未解决的不确定性。
4. 给出更适合什么场景采用正方结论，什么场景必须采纳反方警告。

输出格式：
# 综合结论
# 正方最强论据
# 反方最强论据
# 关键不确定性
# 行动建议"""
        else:
            user_prompt = f"""议题：{state["topic"]}

当前刚完成：第 {completed_round} 轮

外部资料：
{state["search_context"] or "无"}

完整辩论记录：
{_render_history(state["dialogue_history"])}

请输出本轮阶段总结，要求：
1. 点出本轮正方最强论点。
2. 点出本轮反方最强攻击点。
3. 判断当前哪一方在关键问题上更占优，并说明原因。
4. 指出下一轮必须继续追问的关键问题。

输出格式：
阶段总结：
- 正方最强点：
- 反方最强点：
- 当前更占优的一侧及原因：
- 下一轮关键追问："""

        content = await _call_agent(model, MODERATOR_SYSTEM_PROMPT, user_prompt)
        update: dict[str, Any] = {
            "current_turn": completed_round,
            "dialogue_history": [
                {
                    "role": "moderator",
                    "round": completed_round,
                    "content": content,
                }
            ],
        }
        if is_final_round:
            update["final_report"] = content
        return update

    def route_after_moderator(state: DebateState) -> Literal["proponent", END]:
        if state["current_turn"] >= state["max_turns"]:
            return END
        return "proponent"

    builder = StateGraph(DebateState)
    builder.add_node("research", research_node)
    builder.add_node("moderator_opening", moderator_opening_node)
    builder.add_node("proponent", proponent_node)
    builder.add_node("opponent", opponent_node)
    builder.add_node("moderator", moderator_node)

    builder.add_edge(START, "research")
    builder.add_edge("research", "moderator_opening")
    builder.add_edge("moderator_opening", "proponent")
    builder.add_edge("proponent", "opponent")
    builder.add_edge("opponent", "moderator")
    builder.add_conditional_edges("moderator", route_after_moderator)

    return builder.compile()
