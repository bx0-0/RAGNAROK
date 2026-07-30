"""Pydantic schemas for TTS requests."""

from pydantic import BaseModel, Field

try:
    from ..config import TTS_DEFAULT_ENGINE
except ImportError:
    TTS_DEFAULT_ENGINE = 'omnivoice'


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request body."""

    model: str = TTS_DEFAULT_ENGINE            # engine name (registry key)
    input: str = Field(..., min_length=1)       # text to synthesize
    response_format: str = 'wav'                # wav | mp3 (wav default for now)
    voice_instruct: str = ''                    # OmniVoice instruct override

    # Engine-specific tuning params (passed through as kwargs)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    num_step: int = Field(default=16, ge=8, le=32)
    guidance_scale: float = Field(default=2.0, ge=0.1, le=5.0)
    variation: float = Field(default=0.667, ge=0.0, le=1.0)
    seed: int = 7
