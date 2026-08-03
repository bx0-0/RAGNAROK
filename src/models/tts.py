"""Pydantic schemas for TTS requests."""

from pydantic import BaseModel, Field

from src.config import TTS_DEFAULT_ENGINE


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request body."""

    model: str = TTS_DEFAULT_ENGINE            # engine name (registry key)
    input: str = Field(..., min_length=1)       # text to synthesize
    response_format: str = 'wav'                # wav | mp3 (wav default for now)

    # OmniVoice params — send per-request or use defaults below
    voice_instruct: str = 'male, young adult, clear'   # e.g. "male, young adult, british accent"
    num_step: int = Field(default=16, ge=8, le=32)              # diffusion steps
    guidance_scale: float = Field(default=2.0, ge=0.1, le=5.0)

    # Inflect params — send per-request or use defaults below
    variation: float = Field(default=0.667, ge=0.0, le=1.0)     # prosody randomness
    seed: int = 7                                                # reproducibility

    # Shared speed param (OmniVoice \u0026 Inflect both accept it)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
