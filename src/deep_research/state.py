from __future__ import annotations

import operator
from typing import Annotated, Literal

from typing_extensions import TypedDict


DebateRole = Literal["moderator", "proponent", "opponent"]


class DebateTurn(TypedDict):
    role: DebateRole
    round: int
    content: str


class DebateState(TypedDict):
    topic: str
    dialogue_history: Annotated[list[DebateTurn], operator.add]
    current_turn: int
    max_turns: int
    search_context: str
    final_report: str
