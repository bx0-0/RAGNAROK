"""SSE protocol helpers — constants, envelope caching, done-chunk builder."""

from functools import lru_cache

import orjson

# ─── Static SSE bytes ───
_SSE_DONE = b"data: [DONE]\n\n"
_SSE_KEEPALIVE = b"data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":null}]}\n\n"

# Marker for the delta key — orjson outputs this deterministically
_DELTA_KEY = b'"delta":'


@lru_cache(maxsize=16)
def _sse_template_for_model(model: str):
    """Build the SSE JSON envelope once per model.

    Splits *after* `\"delta\":` so prefix ends with the colon and suffix
    starts with `,\"index\":...`. Callers do:
        yield prefix + orjson.dumps(delta) + suffix

    The prefix still contains placeholder id and created values that are
    replaced per-request in make_sse_frames().
    """
    tpl = orjson.dumps({
        "id": "",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"delta": None, "index": 0, "finish_reason": None}],
    })
    # Find the delta key position
    key_pos = tpl.index(_DELTA_KEY) + len(_DELTA_KEY)
    # Advance past `null` to find the comma separator
    suffix_start = tpl.find(b',', key_pos)
    prefix = b"data: " + tpl[:key_pos]
    suffix = tpl[suffix_start:] + b"\n\n"
    return (prefix, suffix)


def make_sse_frames(model: str, request_id_str: str, created: int):
    """Inject real id + timestamp into cached SSE envelope.

    Uses deterministic byte replacement on the placeholder values that
    orjson produces for empty-string id and zero-valued created.
    """
    prefix, suffix = _sse_template_for_model(model)

    # orjson serialises "" → b'""'  (2 bytes)
    _empty_id = b'"id":""'
    _real_id  = b'"id":' + orjson.dumps(request_id_str)

    prefix = prefix.replace(_empty_id, _real_id, 1)
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
