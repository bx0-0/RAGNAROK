"""Pydantic models for /v1/chat/completions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from src.models.shared import ChatMessage
from src.config import THINK_LEVEL_MAP


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]

    @field_validator("messages")
    @classmethod
    def _reject_empty_messages(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v
    model: str | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: str | None = None
    # None = model default (don't send `think` to Ollama)
    thinking: bool | None = None
    # OpenAI/DeepSeek-style: none | minimal | low | medium | high | xhigh
    reasoning_effort: str | None = None

    def to_ollama_payload(
        self,
        active_model: str,
        keep_alive: str,
        ollama_opts: dict[str, Any],
        messages_converted: list[dict],
    ) -> dict[str, Any]:
        """Convert this request into an Ollama chat kwargs dict."""
        # tool_choice: ollama.AsyncClient.chat() ignores this kwarg;
        # the model handles tool selection automatically when tools are present.
        payload = {
            "model": active_model,
            "messages": messages_converted,
            "keep_alive": keep_alive,
            "options": dict(ollama_opts),
        }
        # ── think: top-level Ollama field (not options.thinking) ──
        # reasoning_effort wins over thinking if both are set.
        # Levels are mapped via THINK_LEVEL_MAP (src/config.py) — single edit point.
        # If neither is set, `think` is omitted → Ollama uses the model default.
        if self.reasoning_effort is not None:
            level = self.reasoning_effort.strip().lower()
            payload["think"] = THINK_LEVEL_MAP.get(level, True)
        elif self.thinking is not None:
            payload["think"] = self.thinking
        if self.tools:
            payload["tools"] = [t.model_dump() for t in self.tools]
        return payload
