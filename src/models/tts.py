"""Pydantic schemas for TTS requests."""

from pydantic import BaseModel, Field

from ..config import (
    TTS_DEFAULT_ENGINE,
    TTS_OMNI_INSTRUCT,
    TTS_OMNI_NUM_STEP,
    TTS_OMNI_SPEED,
    TTS_OMNI_GUIDANCE_SCALE,
    TTS_INFLECT_SPEED,
    TTS_INFLECT_VARIATION,
    TTS_INFLECT_SEED,
)


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request body."""

    model: str = TTS_DEFAULT_ENGINE            # engine name (registry key)
    input: str = Field(..., min_length=1)       # text to synthesize
    response_format: str = 'wav'                # wav | mp3 (wav default for now)

    # OmniVoice params (env defaults, override per-request)
    voice_instruct: str = TTS_OMNI_INSTRUCT     # e.g. "male, young adult, british accent"
    num_step: int = Field(default=TTS_OMNI_NUM_STEP, ge=8, le=32)             # diffusion steps
    guidance_scale: float = Field(default=TTS_OMNI_GUIDANCE_SCALE, ge=0.1, le=5.0)

    # Inflect params (env defaults, override per-request)
    variation: float = Field(default=TTS_INFLECT_VARIATION, ge=0.0, le=1.0)   # prosody randomness
    seed: int = TTS_INFLECT_SEED                                               # reproducibility

    # Shared speed param (OmniVoice & Inflect both accept it)
    speed: float = Field(default=TTS_OMNI_SPEED, ge=0.25, le=4.0)
