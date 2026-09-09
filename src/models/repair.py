"""Pydantic model + validation for POST /v1/models/repair."""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


# Ollama model names: <name>[:<tag>].
_NAME_RE = re.compile(r"^[a-zA-Z0-9._\-/]+(:[a-zA-Z0-9._\-]+)?$")


class RepairRequest(BaseModel):
    model: str
    renderer: str | None = None
    parser: str | None = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model must be a non-empty string")
        if not _NAME_RE.match(v):
            raise ValueError(
                f"invalid Ollama model name {v!r} "
                "(expected '<name>[:<tag>]', name/tag use alnum, dot, dash, slash, underscore)"
            )
        if v.count(":") > 1:
            raise ValueError(f"model name may contain at most one ':' tag: {v!r}")
        return v

    @field_validator("renderer", "parser")
    @classmethod
    def _validate_token(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if not re.match(r"^[a-zA-Z0-9._\-]+$", v):
            raise ValueError(
                f"{info.field_name} must be a simple identifier "
                f"(alnum, dot, dash, underscore); got {v!r}"
            )
        return v
