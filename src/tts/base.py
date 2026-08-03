"""Abstract interface for all TTS engines."""

from abc import ABC, abstractmethod


class AbstractTTSEngine(ABC):
    """Base class for TTS backends."""

    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Return raw audio bytes."""

    @abstractmethod
    async def load(self) -> None:
        """Load model into memory. Called lazily on first request."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently in memory."""

    @abstractmethod
    async def unload(self) -> None:
        """Free memory held by this engine."""
