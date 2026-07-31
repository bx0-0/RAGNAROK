"""OmniVoice TTS engine — 600+ languages, voice design via instruct string."""

import asyncio
import gc
import logging
import os
import sys
from typing import Dict, Any

import torch

from .base import AbstractTTSEngine
from . import register as _register

logger = logging.getLogger(__name__)

_MODELS = {
    'omnivoice': 'k2-fsa/OmniVoice',
}

DEFAULT_MODEL = 'omnivoice'


class OmniVoiceEngine(AbstractTTSEngine):
    """Wraps the omnivoice library with lazy-load and GC support."""

    _instance: 'OmniVoiceEngine | None' = None
    _model = None
    _loaded = False

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self._extra_kwargs: Dict[str, Any] = {}

    # ── class-level singleton ───────────────────────────────────────
    @classmethod
    def get(cls, model_name: str = DEFAULT_MODEL, device: str = 'cuda') -> 'OmniVoiceEngine':
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    async def load(self) -> None:
        if self._loaded:
            return

        logger.info(f'Loading OmniVoice model {self.model_name} ...')
        hf_repo = _MODELS.get(self.model_name, self.model_name)

        try:
            # Dynamic import — omnivoice may not be installed
            from omnivoice import OmniVoice  # noqa: F811

            self._model = OmniVoice.from_pretrained(
                hf_repo,
                device_map=self.device,
                dtype=torch.float16,
            )

            # torch.compile for speed (PyTorch >=2.3)
            try:
                self._model = torch.compile(
                    self._model,
                    mode='reduce-overhead',
                    fullgraph=False,
                )
                logger.info('torch.compile enabled for OmniVoice')
            except Exception as e:
                logger.warning(f'torch.compile skipped for OmniVoice: {e}')

            self._loaded = True
            logger.info('OmniVoice loaded successfully')

        except ImportError:
            raise RuntimeError(
                'omnivoice is not installed. Install with: pip install omnivoice torch torchaudio'
            )

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            logger.info('OmniVoice unloaded')

    async def synthesize(self, text: str, **kwargs) -> bytes:
        await self.load()
        assert self._model is not None

        instruct = kwargs.get('voice_instruct', 'male, young adult, clear')
        num_step = int(kwargs.get('num_step', 16))
        guidance = float(kwargs.get('guidance_scale', 2.0))
        speed = float(kwargs.get('speed', 1.0))

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        loop = asyncio.get_event_loop()

        def _gen():
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
                audio = self._model.generate(
                    text=text,
                    instruct=instruct,
                    num_step=num_step,
                    guidance_scale=guidance,
                    speed=speed,
                    postprocess_output=True,
                )
            return audio

        audio = await loop.run_in_executor(None, _gen)
        sample_rate = 24000

        # Return raw PCM wrapped as WAV
        import io
        import struct
        import numpy as np

        pcm = (audio[0].cpu().numpy() * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        # Minimal WAV header
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + len(pcm)))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<I', 16))          # fmt chunk size
        buf.write(struct.pack('<H', 1))           # PCM
        buf.write(struct.pack('<H', 1))           # mono
        buf.write(struct.pack('<I', sample_rate))
        buf.write(struct.pack('<I', sample_rate * 2))
        buf.write(struct.pack('<H', 2))           # block align
        buf.write(struct.pack('<H', 16))          # bits per sample
        buf.write(b'data')
        buf.write(struct.pack('<I', len(pcm)))
        buf.write(pcm)

        return buf.getvalue()


def register_self() -> None:
    _register('omnivoice', OmniVoiceEngine)
