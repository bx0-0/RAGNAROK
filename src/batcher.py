"""StreamBatcher — accumulates text/thinking deltas and emits SSE frames.

Moved verbatim from ``src/streaming.py`` (structural extraction only; no
behavior change). This is the single owner of SSE frame construction for
the streaming path:

* ``flush()`` — emit a buffered text/thinking delta frame.
* ``emit(extra)`` — the SINGLE public path that folds any buffered text into
  a frame carrying *extra* keys (e.g. ``tool_calls``). Streaming.py must NOT
  build SSE frames or call the private ``_delta()`` itself.

Public API (used by streaming.py and tests):
  ``__init__(sfx, efx)``
  ``add(parsed)``            accumulate a parsed chunk's content/thinking
  ``flush()``               -> bytes | None
  ``emit(extra)``           -> bytes | None
  ``reset()``               clear the text/thinking buffer (retry boundary)
  ``dirty()``               -> bool
  ``content_len()``         -> int
  ``thinking_len()``        -> int
  ``tool_call_index``       int (next index to assign to a new tool call)

Private (never call from outside this module):
  ``_delta(extra)``         build the raw delta dict (role/content/thinking + extra)
"""

import orjson


class StreamBatcher:
    """Accumulates reasoning/content deltas and emits batched SSE frames.

    Owns the "first chunk gets role" invariant and the batching buffer so the
    main loop never has to juggle those locals itself. `tool_call_index` is
    owned here too — the next index to assign to a new tool call.
    """

    def __init__(self, sfx: bytes, efx: bytes):
        self._sfx = sfx
        self._efx = efx
        self._first = True
        self._content: list = []
        self._thinking: list = []
        self.tool_call_index = 0

    # -- introspection ------------------------------------------------------
    def _delta(self, extra: dict | None = None) -> dict:
        delta = {}
        if self._first:
            delta["role"] = "assistant"
            self._first = False
        if self._thinking:
            delta["reasoning_content"] = "".join(self._thinking)
        if self._content:
            delta["content"] = "".join(self._content)
        if extra:
            delta.update(extra)
        return delta

    def dirty(self) -> bool:
        """True if there is unflushed reasoning/content buffered."""
        return bool(self._content or self._thinking)

    def content_len(self) -> int:
        return len("".join(self._content))

    def thinking_len(self) -> int:
        return len("".join(self._thinking))

    def reset(self) -> None:
        """Clear buffered text/thinking (keeps the first-frame flag and the
        tool_call_index). Called at the start of each retry attempt."""
        self._content.clear()
        self._thinking.clear()

    # -- emission -----------------------------------------------------------
    def flush(self) -> bytes | None:
        """Emit a buffered delta frame and clear the buffer. None if empty."""
        if not self._content and not self._thinking:
            return None
        delta = self._delta()
        self._content.clear()
        self._thinking.clear()
        return self._sfx + orjson.dumps(delta) + self._efx

    def emit(self, extra: dict) -> bytes | None:
        """Emit a delta frame carrying *extra* keys (e.g. tool_calls).

        Any buffered text/thinking is folded into the same frame first, then
        the buffer is cleared — a tool call and its preceding text ship together.
        """
        if not self._content and not self._thinking and not extra:
            return None
        delta = self._delta(extra)
        self._content.clear()
        self._thinking.clear()
        return self._sfx + orjson.dumps(delta) + self._efx

    def add(self, parsed) -> None:
        """Accumulate a parsed chunk's text/thinking into the buffer.

        ``parsed`` is duck-typed: it needs ``.content`` (str) and ``.thinking``
        (str) — set by the parser's ``_ParsedChunk``. Tool calls are NOT
        accumulated — they force an immediate flush via ``emit``; the caller
        handles those separately.
        """
        if parsed.content:
            self._content.append(parsed.content)
        if parsed.thinking:
            self._thinking.append(parsed.thinking)
