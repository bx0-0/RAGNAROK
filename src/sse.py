"""SSE protocol helpers — constants, envelope caching, done-chunk builder."""

from functools import lru_cache

import orjson

# ─── Static SSE bytes ───
_SSE_DONE = b"data: [DONE]\n\n"
_SSE_KEEPALIVE = b": ping\n\n"

# Marker used inside cached template to be replaced at runtime
_SSE_MARKER_ID = "\xffID"


@lru_cache(maxsize=16)
def _sse_template_for_model(model: str):
    """Build the SSE JSON envelope once per model with unique string markers.

    Returns (prefix_bytes, suffix_bytes) ready for:
        yield prefix + orjson.dumps(delta) + suffix
    """
    tpl = orjson.dumps({
        "id": _SSE_MARKER_ID,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"delta": "__DELTA__", "index": 0, "finish_reason": None}],
    })
    _marker = b'"__DELTA__"'

    _dpos = tpl.index(_marker)
    prefix = tpl[:_dpos]
    suffix = tpl[_dpos + len(_marker):]
    return (b"data: " + prefix, suffix + b"\n\n")


def make_sse_frames(model: str, request_id_str: str, created: int):
    """Inject real id + timestamp into cached SSE envelope."""
    prefix, suffix = _sse_template_for_model(model)
    _id_placeholder = f'"{_SSE_MARKER_ID}"'.encode()
    _real_id = f'"{request_id_str}"'.encode()
    prefix = prefix.replace(_id_placeholder, _real_id, 1)
    prefix = prefix.replace(b'"created":0', f'"created":{created}'.encode(), 1)
    return (prefix, suffix)


def build_done_chunk(request_id_str: str, created: int, model: str,
                     has_tool_calls: bool, prompt_tokens: int, completion_tokens: int) -> bytes:
    """Build the final usage/finish chunk (identical in graceful path and finally)."""
    finish = "tool_calls" if has_tool_calls else "stop"
    return (
        b"data: " + orjson.dumps({
            "id": request_id_str,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }) + b"\n\n"
    )
