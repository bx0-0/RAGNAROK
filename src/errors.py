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
    content=orjson.dumps({
        "error": {"message": "Invalid JSON", "type": "invalid_request_error"},
    }),
    media_type="application/json",
)


def build_sse_error_frame(message: str, error_type: str = "server_error") -> bytes:
    """Build a single SSE data frame containing an error payload."""
    return b"data: " + orjson.dumps({"error": {"message": message, "type": error_type}}) + b"\n\n"

# ─── Repairable Ollama runtime errors ───────────────────────────────────────
# Known Ollama chat-template/runtime incompatibilities that an explicit
# runtime repair (POST /v1/models/repair) can resolve. RAGNAROK never repairs
# automatically — it only surfaces these as "repair available" hints.
_REPAIRABLE_HINTS = (
    "system message must be at the beginning",
    "must be at the beginning",
    "unexpected role",
    "chat template",
    "template error",
    "invalid renderer",
    "invalid parser",
    "unknown renderer",
    "unknown parser",
)


def is_repairable_error(message: str) -> bool:
    """True if an Ollama error message looks like a runtime/renderer/parser
    incompatibility that an explicit repair could fix. Case-insensitive."""
    m = (message or "").lower()
    return any(h in m for h in _REPAIRABLE_HINTS)


def build_repair_available_response(detail: str) -> Response:
    """500 error informing the client that an explicit repair is available.

    RAGNAROK does NOT auto-repair or guess renderer/parser; it points the
    client at the repair endpoint.
    """
    return Response(
        status_code=500,
        content=orjson.dumps({
            "error": {
                "message": "Model runtime configuration may be incompatible with this request.",
                "type": "model_runtime_incompatible",
                "code": "REPAIR_AVAILABLE",
                "detail": (detail or "")[:120],
                "repair_available": True,
                "repair_endpoint": "/v1/models/repair",
            }
        }),
        media_type="application/json",
    )


def build_sse_repair_available_frame(detail: str) -> bytes:
    """SSE error frame variant of build_repair_available_response."""
    payload = {
        "error": {
            "message": "Model runtime configuration may be incompatible with this request.",
            "type": "model_runtime_incompatible",
            "code": "REPAIR_AVAILABLE",
            "detail": (detail or "")[:120],
            "repair_available": True,
            "repair_endpoint": "/v1/models/repair",
        }
    }
    return b"data: " + orjson.dumps(payload) + b"\n\n"
