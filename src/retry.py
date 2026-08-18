"""RetryPolicy — centralized retry configuration and decision logic.

Previously these lived as scattered module constants / env reads in
``streaming.py``:

* ``MAX_RETRIES = 2``           → max retries (3 total attempts)
* ``_should_retry_empty()``     → env RETRY_ON_EMPTY (default "False")
* ``asyncio.sleep(2)``          → crashed-mid-stream backoff (hardcoded)
* ``asyncio.sleep(1)``          → empty-stream backoff (hardcoded)

Behavior is preserved exactly: same default values, same decision
semantics, same "count then decide" order (``retry_count`` is incremented
before the retry check in every branch, as before).

The policy is deliberately **stateless** — callers own the ``retry_count``
counter and pass it in, so no retry bookkeeping has moved out of
``streaming.py``; only the configuration and the boolean decisions live here.

Tests can construct ``RetryPolicy`` with small backoff values and a
different ``max_retries`` to exercise the loop without waiting on real
delays or hammering the upstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = ("true", "1", "yes")


def _truthy(value: str | None) -> bool:
    """Mirror of the previous inline env check: lowercase, case-insensitive,
    accepts ``true`` / ``1`` / ``yes``; anything else (including ``None``)
    is False.
    """
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


@dataclass(frozen=True)
class RetryPolicy:
    """Immutable retry configuration + decision helpers.

    ``max_retries`` is the number of *retries* after the initial attempt,
    matching the previous ``MAX_RETRIES`` constant (total attempts =
    ``max_retries + 1``).
    """

    max_retries: int = 2
    crashed_backoff_s: float = 2.0
    empty_backoff_s: float = 1.0
    retry_on_empty: bool = False

    @classmethod
    def default(cls) -> "RetryPolicy":
        """Build a policy from the environment with the historical defaults.

        ``RETRY_ON_EMPTY`` is the only env knob; the two backoffs are
        constants today but are exposed so tests can shrink them.
        """
        return cls(
            max_retries=2,
            crashed_backoff_s=2.0,
            empty_backoff_s=1.0,
            retry_on_empty=_truthy(os.environ.get("RETRY_ON_EMPTY", "False")),
        )

    # -- decision helpers ---------------------------------------------------
    #
    # All helpers take ``retry_count`` — the *new* value after the caller
    # has incremented it — to preserve the exact "increment-then-decide"
    # order the old inline code used.

    def should_retry_crashed(self, retry_count: int) -> bool:
        """Died-mid-stream: always retry until exhausted (env-independent)."""
        return retry_count <= self.max_retries

    def should_retry_empty(self, retry_count: int) -> bool:
        """Empty stream: retry only if enabled AND not exhausted."""
        return self.retry_on_empty and retry_count <= self.max_retries

    def next_delay(self, kind: str, retry_count: int) -> float:
        """Backoff seconds for the given failure kind.

        ``kind`` is ``"crashed"`` or ``"empty"``. Raises ``ValueError`` for
        unknown kinds — the only two callers in ``streaming.py`` are typed,
        so a bad kind is a programmer error, not a runtime condition.
        """
        if kind == "crashed":
            return self.crashed_backoff_s
        if kind == "empty":
            return self.empty_backoff_s
        raise ValueError(f"unknown retry kind: {kind!r}")
