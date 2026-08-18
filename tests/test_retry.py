"""RetryPolicy unit + integration tests.

Unit tests cover the policy in isolation (pure, no I/O). Integration tests
drive the REAL ``stream_generator`` with a fake Ollama client to verify the
loop honours the policy: retry count, exhaustion, env-driven empty-stream
behavior, and non-retryable failures.

These preserve the exact pre-existing retry behavior:
* died-mid-stream  -> always retries (env-independent), 3 total attempts by default
* empty-stream      -> retries only when RETRY_ON_EMPTY is truthy
* done-with-tokens  -> success, no retry
* ResponseError     -> no retry
"""
import asyncio
from types import SimpleNamespace as NS

import pytest

from src.retry import RetryPolicy, _truthy


# ── pure unit tests ──────────────────────────────────────────────────────

class TestPolicyDefaults:

    def test_default_values_match_legacy_constants(self):
        p = RetryPolicy(max_retries=2, crashed_backoff_s=2.0,
                        empty_backoff_s=1.0, retry_on_empty=False)
        assert p.max_retries == 2          # legacy MAX_RETRIES
        assert p.crashed_backoff_s == 2.0  # legacy sleep(2)
        assert p.empty_backoff_s == 1.0    # legacy sleep(1)
        assert p.retry_on_empty is False   # legacy RETRY_ON_EMPTY default

    def test_frozen(self):
        p = RetryPolicy()
        with pytest.raises(Exception):
            p.max_retries = 9  # type: ignore[misc]


class TestTruthiness:

    @pytest.mark.parametrize("val,expected", [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("no", False),
        ("True", True),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("  TRUE  ", True),
    ])
    def test_truthy(self, val, expected):
        assert _truthy(val) is expected


class TestDecisions:

    def setup_method(self):
        self.p = RetryPolicy(max_retries=2, retry_on_empty=True)

    def test_crashed_retries_until_exhausted(self):
        # retry_count is the value AFTER incrementing (caller semantics)
        assert self.p.should_retry_crashed(1) is True
        assert self.p.should_retry_crashed(2) is True
        assert self.p.should_retry_crashed(3) is False   # 3 > max_retries=2
        assert self.p.should_retry_crashed(4) is False

    def test_empty_respects_toggle(self):
        on = RetryPolicy(max_retries=2, retry_on_empty=True)
        off = RetryPolicy(max_retries=2, retry_on_empty=False)
        assert on.should_retry_empty(1) is True
        assert on.should_retry_empty(2) is True
        assert on.should_retry_empty(3) is False
        assert off.should_retry_empty(1) is False        # disabled regardless
        assert off.should_retry_empty(2) is False

    def test_next_delay_by_kind(self):
        p = RetryPolicy(crashed_backoff_s=2.5, empty_backoff_s=0.75)
        assert p.next_delay("crashed", 1) == 2.5
        assert p.next_delay("empty", 1) == 0.75

    def test_next_delay_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            RetryPolicy().next_delay("bogus", 1)


class TestEnvDefault:
    """default() reads RETRY_ON_EMPTY with the historical 'False' default."""

    def test_default_reads_env(self, monkeypatch):
        monkeypatch.setenv("RETRY_ON_EMPTY", "yes")
        assert RetryPolicy.default().retry_on_empty is True

        monkeypatch.setenv("RETRY_ON_EMPTY", "nope")
        assert RetryPolicy.default().retry_on_empty is False

    def test_default_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("RETRY_ON_EMPTY", raising=False)
        assert RetryPolicy.default().retry_on_empty is False


# ── integration: drive the REAL stream_generator ─────────────────────────

def _raw_chunk(content=None, done=False, prompt=0, evalc=0, reason="stop"):
    return NS(message=NS(content=content, thinking="", tool_calls=None),
              done=done, prompt_eval_count=prompt, eval_count=evalc,
              done_reason=reason)


class CountingClient:
    """Fake Ollama client that records chat() call count and yields canned
    chunks. ``raise_exc`` simulates a stream error."""

    def __init__(self, chunks, raise_exc=None):
        self._chunks = chunks
        self._raise = raise_exc
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if self._raise:
            raise self._raise
        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()


def _drive(client, monkeypatch_retry):
    import src.streaming as st
    if monkeypatch_retry is not None:
        st._RETRY = monkeypatch_retry
    state = NS(http_client=client, semaphore=asyncio.Semaphore(2))
    sfx, efx = st.make_sse_frames("m", "chatcmpl-req", 0)

    async def run():
        gen = st.stream_generator(
            state, "req", {"model": "m", "messages": []},
            asyncio.get_event_loop().time(), "chatcmpl-req", 0, "m", 300,
            sfx, efx,
        )
        return b"".join([f async for f in gen])

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()


class TestLoopHonorsPolicy:

    def test_empty_stream_retries_when_enabled(self):
        """3 attempts total (initial + 2 retries) when RETRY_ON_EMPTY on.

        Fake yields a clean empty stream (no done, no tokens) every call.
        """
        fast = RetryPolicy(max_retries=2, crashed_backoff_s=0.0,
                           empty_backoff_s=0.0, retry_on_empty=True)
        client = CountingClient([_raw_chunk(content="")])
        _drive(client, fast)
        assert client.calls == 3  # exhausted -> 1 initial + 2 retries

    def test_empty_stream_no_retry_when_disabled(self):
        """Legacy default: empty stream is NOT retried."""
        fast = RetryPolicy(max_retries=2, crashed_backoff_s=0.0,
                           empty_backoff_s=0.0, retry_on_empty=False)
        client = CountingClient([_raw_chunk(content="")])
        _drive(client, fast)
        assert client.calls == 1  # no retry

    def test_done_with_tokens_no_retry(self):
        """A normal done stream is a single successful attempt."""
        fast = RetryPolicy(max_retries=2, crashed_backoff_s=0.0,
                           empty_backoff_s=0.0, retry_on_empty=True)
        client = CountingClient([_raw_chunk(content="hi", done=True,
                                            prompt=3, evalc=5)])
        out = _drive(client, fast)
        assert client.calls == 1
        assert b'"finish_reason":"stop"' in out
