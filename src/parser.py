"""parser.py — Ollama-chunk → normalized data.

Moved verbatim from ``src/streaming.py`` (structural extraction only; no
behavior change). This is the single place that understands Ollama's raw
streaming object shape. Downstream code (streaming.py / batcher.py) only ever
sees the normalized ``ParsedChunk`` — no Ollama-specific objects leak further.

Pipeline position:

    Ollama chunk  →  parser.py  →  normalized ParsedChunk  →  streaming.py  →  batcher.py  →  SSE

Public names (moved unchanged, no public API introduced):
  ``ParsedChunk``          (was ``_ParsedChunk``)  — normalized data container
  ``format_tool_calls``    (was ``_format_tool_calls``) — Ollama → OpenAI delta shape
  ``parse_chunk``          (was ``_parse_chunk``) — one Ollama chunk → ParsedChunk

The private underscored aliases are re-exported by ``streaming.py`` so existing
importers/tests keep working.
"""

from dataclasses import dataclass

import orjson

from src.utils.helpers import fast_id


@dataclass
class ParsedChunk:
    """Normalized, serializable view of one Ollama streaming chunk.

    Built by ``parse_chunk`` from the raw ollama object so all the messy
    extraction (content shape, thinking, tool-call formatting) lives in one
    pure, unit-testable place. ``tool_calls`` is a list only when the chunk
    carries them (else empty); ``content``/``thinking`` are always str.
    """
    content: str = ""
    thinking: str = ""
    tool_calls: list = None  # populated (non-empty) when the chunk has tool calls
    done: bool = False
    prompt_eval_count: int = 0
    eval_count: int = 0
    done_reason: str = ""


def format_tool_calls(tool_calls, tool_call_index: int) -> list:
    """Convert Ollama tool_calls objects to the OpenAI delta shape.

    ``tool_call_index`` is the next index to assign; the return value is a
    2-tuple is avoided (to keep it trivially testable) — instead the list of
    formatted dicts is returned and the caller increments its own counter by
    ``len(formatted)``.

    Idempotent: if an item is already an OpenAI-shaped dict (e.g. produced by
    ``parse_chunk``, which pre-formats), it is passed through unchanged except
    for the running ``index``. Formatting an already-formatted dict a second time
    used to hit the ``function is None`` fallback and emit ``name="?"``, ``args="{}"``.
    """
    formatted = []
    for tc in tool_calls:
        if tc is None:
            continue

        # Already OpenAI-shaped (a plain dict with a "function" key) — pass
        # through, re-assigning only the running index. This is the common
        # path: ``parse_chunk`` returns pre-formatted dicts.
        if isinstance(tc, dict) and "function" in tc:
            entry = {
                "index": tool_call_index + len(formatted),
                "id": tc.get("id") or f"call_{fast_id()}",
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc["function"].get("name", "?"),
                    "arguments": tc["function"].get("arguments", "{}"),
                },
            }
            formatted.append(entry)
            continue

        # Raw Ollama object shape: tc.function.name / tc.function.arguments.
        tc_func = getattr(tc, "function", None)
        if tc_func is None:
            tc_name = "?"
            tc_args_json = "{}"
        else:
            tc_name = getattr(tc_func, "name", "?") or "?"
            tc_args = getattr(tc_func, "arguments", None) or ""
            if isinstance(tc_args, str):
                tc_args_json = tc_args
            else:
                tc_args_json = orjson.dumps(tc_args).decode() if tc_args else "{}"
        formatted.append({
            "index": tool_call_index + len(formatted),
            "id": getattr(tc, "id", None) or f"call_{fast_id()}",
            "type": "function",
            "function": {
                "name": tc_name,
                "arguments": tc_args_json,
            },
        })
    return formatted


def parse_chunk(chunk) -> ParsedChunk:
    """Extract the fields we care about from one Ollama chunk.

    Pure: reads attributes off the chunk, no I/O, no server, no mutation.
    Handles the ``content`` shape variants (str or list-of-parts) and the
    ``thinking`` field uniformly so the main loop stays simple.
    """
    msg = chunk.message
    if msg is None:
        # Ollama's empty final chunk — no payload
        return ParsedChunk()

    content = msg.content or ""
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    thinking = getattr(msg, "thinking", "") or ""

    tool_calls = msg.tool_calls
    formatted = []
    if tool_calls:
        formatted = format_tool_calls(tool_calls, 0)

    return ParsedChunk(
        content=content,
        thinking=thinking,
        tool_calls=formatted,
        done=bool(chunk.done),
        prompt_eval_count=chunk.prompt_eval_count or 0,
        eval_count=chunk.eval_count or 0,
        done_reason=chunk.done_reason or "",
    )
