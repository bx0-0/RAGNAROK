"""Garbage Collector for loaded models.

Tracks all loaded artifacts (Ollama model, TTS engines) and evicts
idle ones after a configurable timeout so Kaggle/Colab RAM stays free.

Features:
- Per-engine idle timeout (default 10 min)
- Automatic background sweep every 60 s
- Manual unload via DELETE /v1/tts/unload or /v1/models/unload
- Health endpoint reports resident models and memory pressure
"""

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class _ModelSlot:
    """Tracks one loaded model / engine."""
    name: str
    kind: str           # 'ollama' | 'tts_omnivoice' | 'tts_inflect' | ...
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    unload_fn: Optional[Callable] = None
    is_loaded_fn: Optional[Callable] = None


class ModelGC:
    """Singleton garbage collector for transient model loading."""

    _instance: Optional['ModelGC'] = None

    def __init__(self, idle_timeout: float = 600.0, sweep_interval: float = 60.0):
        """
        Args:
            idle_timeout: Seconds before an unused model is evicted.
                0 or negative = never auto-evict (default Kaggle behaviour).
            sweep_interval: How often the background task checks.
        """
        self.idle_timeout = idle_timeout
        self.sweep_interval = sweep_interval
        self._slots: Dict[str, _ModelSlot] = {}
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ── singleton ────────────────────────────────────────────────
    @classmethod
    def get(cls, idle_timeout: float = 600.0, sweep_interval: float = 60.0) -> 'ModelGC':
        if cls._instance is None:
            cls._instance = cls(idle_timeout=idle_timeout, sweep_interval=sweep_interval)
        return cls._instance

    # ── lifecycle ────────────────────────────────────────────────
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._sweep_loop(), name='model-gc-sweep')
        logger.info(
            f'ModelGC started  (idle_timeout={self.idle_timeout:.0f}s, '
            f'sweep_interval={self.sweep_interval:.0f}s)'
        )

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Force unload everything
        await self.unload_all()

    # ── registration ─────────────────────────────────────────────
    async def register(
        self,
        name: str,
        kind: str,
        unload_fn: Callable,
        is_loaded_fn: Optional[Callable] = None,
    ) -> None:
        async with self._lock:
            existing = self._slots.get(name)
            if existing:
                existing.unload_fn = unload_fn
                if is_loaded_fn:
                    existing.is_loaded_fn = is_loaded_fn
            else:
                self._slots[name] = _ModelSlot(
                    name=name,
                    kind=kind,
                    unload_fn=unload_fn,
                    is_loaded_fn=is_loaded_fn,
                )

    # ── touch ────────────────────────────────────────────────────
    def touch(self, name: str) -> None:
        slot = self._slots.get(name)
        if slot:
            slot.last_used = time.monotonic()

    # ── snapshot for /health ─────────────────────────────────────
    async def status(self) -> Dict[str, Any]:
        """Return current state of all tracked models."""
        now = time.monotonic()
        info: Dict[str, Any] = {}
        async with self._lock:
            for key, s in self._slots.items():
                age_idle = round(now - s.last_used, 1)
                age_total = round(now - s.loaded_at, 1)
                loaded = True
                if s.is_loaded_fn:
                    try:
                        loaded = bool(s.is_loaded_fn())
                    except Exception:
                        loaded = None
                info[key] = {
                    'kind': s.kind,
                    'loaded': loaded,
                    'idle_seconds': age_idle,
                    'total_resident_seconds': age_total,
                    'auto_evict_after_s': self.idle_timeout if self.idle_timeout > 0 else None,
                }
        return info

    # ── unload helpers ───────────────────────────────────────────
    async def unload(self, name: str) -> bool:
        """Unload a specific model by name. Returns True if something was freed."""
        async with self._lock:
            slot = self._slots.get(name)
            if slot and slot.unload_fn:
                try:
                    if asyncio.iscoroutinefunction(slot.unload_fn):
                        await slot.unload_fn()
                    else:
                        slot.unload_fn()
                except Exception as exc:
                    logger.error(f'ModelGC unload error for {name}: {exc}')
                finally:
                    del self._slots[name]
                    gc.collect()
                    return True
        return False

    async def unload_all(self) -> int:
        """Unload everything. Returns number of models freed."""
        names = list(self._slots.keys())
        count = 0
        for n in names:
            if await self.unload(n):
                count += 1
        return count

    # ── background sweep ────────────────────────────────────────
    async def _sweep_loop(self) -> None:
        """Periodically evict idle models."""
        while True:
            try:
                await asyncio.sleep(self.sweep_interval)
                if self.idle_timeout <= 0:
                    continue
                now = time.monotonic()
                async with self._lock:
                    to_evict = [
                        name
                        for name, s in self._slots.items()
                        if (now - s.last_used) > self.idle_timeout
                    ]
                for name in to_evict:
                    logger.info(f'ModelGC evicting idle model: {name}')
                    await self.unload(name)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f'ModelGC sweep error: {exc}')
