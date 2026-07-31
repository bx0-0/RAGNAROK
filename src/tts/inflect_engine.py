"""Inflect v2 TTS engine — English-only, ultra-small (~16MB Nano / ~38MB Micro).

Installed from:  git+https://github.com/owenawsong/Inflect.git

This file is a thin wrapper that registers itself with the TTS registry.
If the library is not installed, registration is skipped silently and
the engine is unavailable at runtime (route returns 501).
"""

import asyncio
import gc
import logging
from typing import Dict, Any

from .base import AbstractTTSEngine
from . import register as _register

logger = logging.getLogger(__name__)


class InflectEngine(AbstractTTSEngine):
    """Wraps the InflectTTS library (owenawsong/Inflect v2)."""

    _instance: 'InflectEngine | None' = None
    _model = None
    _loaded = False

    def __init__(self, variant: str = 'nano'):
        self.variant = variant  # 'nano' or 'micro'
        self._extra_kwargs: Dict[str, Any] = {}

    @classmethod
    def get(cls, variant: str = 'nano') -> 'InflectEngine':
        if cls._instance is None or cls._instance.variant != variant:
            cls._instance = cls(variant=variant)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    async def load(self) -> None:
        if self._loaded:
            return

        logger.info(f'Loading InflectTTS ({self.variant}) ...')

        try:
            from inference import InflectTTS  # noqa: F811

            # Auto-download model from HF Hub if not present
            import os
            cache_dir = os.path.expanduser('~/.cache/inflect_tts')
            repo_id = 'owensong/Inflect-Micro-v2' if self.variant == 'micro' else 'owensong/Inflect-Nano-v2'

            if not os.path.isdir(cache_dir):
                logger.info(f'Downloading Inflect model from {repo_id} ...')
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=repo_id, local_dir=cache_dir)
                except ImportError:
                    raise RuntimeError(
                        'huggingface_hub is not installed. Install with: pip install huggingface_hub'
                    )

            self._model = InflectTTS(cache_dir, device='cpu')
            self._loaded = True
            logger.info(f'InflectTTS loaded ({self.variant})')
        except ImportError as exc:
            if 'inference' in str(exc):
                raise RuntimeError(
                    'InflectTTS is not installed. Install with: pip install inference'
                )
            raise

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            gc.collect()
            logger.info('InflectTTS unloaded')

    async def synthesize(self, text: str, **kwargs) -> bytes:
        await self.load()
        assert self._model is not None

        speed = float(kwargs.get('speed', 1.0))
        variation = float(kwargs.get('variation', 0.667))
        seed = int(kwargs.get('seed', 7))

        loop = asyncio.get_event_loop()
        import io

        def _gen():
            buf = io.BytesIO()
            self._model.save(
                text,
                buf,
                speed=speed,
                variation=variation,
                seed=seed,
            )
            return buf.getvalue()

        try:
            return await loop.run_in_executor(None, _gen)
        except Exception:
            # Fallback: save to temp file then read
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            self._model.save(
                text,
                tmp.name,
                speed=speed,
                variation=variation,
                seed=seed,
            )
            with open(tmp.name, 'rb') as f:
                data = f.read()
            import os
            os.unlink(tmp.name)
            return data


def register_self() -> None:
    _register('inflect', InflectEngine)
