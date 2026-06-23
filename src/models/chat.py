"""Pydantic models for /v1/chat/completions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models.shared import ChatMessage


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
    thinking: bool = False

    def to_ollama_payload(
        self,
        active_model: str,
        keep_alive: str,
        ollama_opts: dict[str, Any],
        messages_converted: list[dict],
    ) -> dict[str, Any]:
        """Convert this request into an Ollama chat kwargs dict."""
        payload = {
            "model": active_model,
            "messages": messages_converted,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {**ollama_opts, "thinking": {"enabled": self.thinking}},
        }
        if self.tools:
            payload["tools"] = [t.model_dump() for t in self.tools]
        if self.tool_choice:
            payload["tool_choice"] = self.tool_choice
        return payload
