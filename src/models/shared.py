"""Pydantic models — request/response schemas for OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str = ""
    arguments: dict | str = {}


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
