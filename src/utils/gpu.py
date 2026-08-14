"""GPU memory utilities for VRAM guards."""

import logging

logger = logging.getLogger(__name__)


class InsufficientVRAMError(Exception):
    """Raised when GPU memory is below the required threshold."""
    def __init__(self, free_gb: float, required_gb: float):
        self.free_gb = free_gb
        self.required_gb = required_gb
        super().__init__(f'GPU memory insufficient ({free_gb:.2f} GB free < {required_gb} GB)')


def get_free_vram_gb(device_idx: int = 0) -> float:
    """Return free VRAM in GB for the given device."""
    try:
        import torch
    except Exception:
        logger.warning('torch not available — cannot query VRAM')
        return float('inf')

    if not torch.cuda.is_available():
        return float('inf')

    try:
        total = torch.cuda.get_device_properties(device_idx).total_memory
        reserved = torch.cuda.memory_reserved(device_idx)
        free = total - reserved
        return free / (1024 ** 3)
    except Exception as exc:
        logger.error(f'Failed to query GPU memory: {exc}')
        return float('inf')


async def _unload_ollama_models() -> None:
    """Unload all loaded Ollama models via the API."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            # Get loaded models
            resp = await client.get('http://localhost:11434/api/tags')
            models = resp.json().get('models', [])
            if not models:
                logger.info('No Ollama models loaded')
                return

            for m in models:
                name = m.get('name')
                if name:
                    logger.info(f'Unloading Ollama model: {name}')
                    await client.post(
                        'http://localhost:11434/api/unload',
                        json={'model': name}
                    )
    except Exception as e:
        logger.error(f'Ollama unload failed: {e}')


async def ensure_free_vram(min_gb: float, gc) -> None:
    """Ensure at least min_gb VRAM is free. Unloads Ollama models + TTS engines if needed.

    Args:
        min_gb: Minimum free VRAM in GB required
        gc: ModelGC instance for unloading TTS engines

    Raises:
        InsufficientVRAMError: If VRAM remains below threshold after cleanup
    """
    free_gb = get_free_vram_gb()
    if free_gb >= min_gb:
        return

    logger.warning(
        f'GPU memory low: {free_gb:.2f} GB free < {min_gb} GB. Unloading LLMs and TTS engines...'
    )

    # 1. Unload Ollama models (big VRAM consumer)
    await _unload_ollama_models()

    # 2. Unload TTS engines registered with GC
    await gc.unload_all()

    # Re-check
    free_gb2 = get_free_vram_gb()
    if free_gb2 < min_gb:
        raise InsufficientVRAMError(free_gb2, min_gb)
