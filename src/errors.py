"""Centralized error responses for both streaming (SSE yields) and non-streaming (FastAPI Response)."""

import orjson
from fastapi.responses import Response


# ─── Pre-built FastAPI Responses (singleton, created once at import) ───
_RATE_LIMIT_RESPONSE = Response(
    status_code=429,
    content=orjson.dumps({
        "error": {"message": "Server is busy. Try again shortly.", "type": "rate_limit_error"},
    }),
    media_type="application/json",
)

_BAD_JSON_RESPONSE = Response(
    status_code=400,
    content=b'{"error":"Invalid JSON"}',
    media_type="application/json",
)


def build_sse_error_frame(message: str, error_type: str = "server_error") -> bytes:
    """Build a single SSE data frame containing an error payload."""
    return b"data: " + orjson.dumps({"error": {"message": message, "type": error_type}}) + b"\n\n"
