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


def build_chat_kwargs(payload: dict[str, Any], stream: bool = False) -> dict[str, Any]:
    """Build the ollama.AsyncClient.chat() kwargs dict from an Ollama payload.

    Single source of truth so the non-stream (routes/chat.py) and stream
    (streaming.py) paths cannot drift apart. Only keys present in ``payload``
    are forwarded; ``stream`` is always set.

    NB: ``tool_choice`` is intentionally NOT forwarded — the ollama client
    does not accept it, and Ollama defaults to "auto" when tools are present.
    """
    kwargs: dict[str, Any] = {
        "model": payload["model"],
        "messages": payload["messages"],
        "stream": stream,
    }
    if payload.get("keep_alive"):
        kwargs["keep_alive"] = payload["keep_alive"]
    if payload.get("options"):
        kwargs["options"] = payload["options"]
    if "think" in payload:
        kwargs["think"] = payload["think"]
    if payload.get("tools"):
        kwargs["tools"] = payload["tools"]
    return kwargs
