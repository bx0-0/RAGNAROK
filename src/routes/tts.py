"""TTS endpoint — OpenAI-compatible POST /v1/audio/speech.

Uses the plugin registry so new engines are zero-code to add.
Registered engine is lazily loaded on first use and tracked by ModelGC.
"""

import asyncio
import logging
import time
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.models.tts import SpeechRequest
from src.gc import ModelGC
from ..tts import get_engine_class, available_engines
from ..logging import logger
from ...config import TTS_ENABLED, TTS_MAX_CHARS

router = APIRouter(prefix='/v1/audio')
_gc = ModelGC.get()


async def _get_or_load_engine(name: str):
    """Return an engine instance, loading it lazily if needed."""
    cls = get_engine_class(name)
    eng = cls.get()

    # Register with GC on first use
    await _gc.register(
        name=name,
        kind=f'tts_{name}',
        unload_fn=eng.unload,
        is_loaded_fn=lambda: eng.is_loaded,
    )

    if not eng.is_loaded:
        await eng.load()

    return eng


@router.post('/speech')
async def speech(request: SpeechRequest, req: Request):
    """Generate speech from text. OpenAI-compatible."""
    if not TTS_ENABLED:
        return Response(
            content='TTS is disabled. Set TTS_ENABLED=true',
            status_code=HTTPStatus.BAD_GATEWAY,
            media_type='text/plain',
        )

    if len(request.input) > TTS_MAX_CHARS:
        return Response(
            content=f'Text too long ({len(request.input)} chars). Max: {TTS_MAX_CHARS}',
            status_code=HTTPStatus.BAD_REQUEST,
            media_type='text/plain',
        )

    t0 = time.monotonic()

    # Route to the right engine
    engine_name = request.model
    if engine_name not in available_engines():
        return Response(
            content=f'Unknown TTS engine "{engine_name}". Available: {available_engines()}',
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            media_type='text/plain',
        )

    try:
        eng = await _get_or_load_engine(engine_name)
    except RuntimeError as exc:
        return Response(
            content=str(exc),
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            media_type='text/plain',
        )

    # Build kwargs passed to engine.synthesize()
    kwargs = {
        'speed': request.speed,
    }
    if request.voice_instruct:
        kwargs['voice_instruct'] = request.voice_instruct

    # Engine-specific params
    if engine_name == 'omnivoice':
        kwargs['num_step'] = request.num_step
        kwargs['guidance_scale'] = request.guidance_scale
    elif engine_name == 'inflect':
        kwargs['variation'] = request.variation
        kwargs['seed'] = request.seed

    try:
        audio_bytes = await eng.synthesize(request.input, **kwargs)
    except Exception as exc:
        logger.exception(f'TTS synthesis failed: {exc}')
        return Response(
            content=f'TTS generation failed: {exc}',
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            media_type='text/plain',
        )

    # Touch GC so engine stays alive while in use
    _gc.touch(engine_name)

    elapsed = round(time.monotonic() - t0, 2)
    logger.info(
        f'TTS {engine_name}: {len(request.input)} chars → '
        f'{len(audio_bytes):,} bytes in {elapsed}s'
    )

    return Response(
        content=audio_bytes,
        media_type='audio/wav',
        headers={
            'X-TTS-Engine': engine_name,
            'X-TTS-Elapsed-S': str(elapsed),
        },
    )


@router.post('/unload')
async def unload_tts(req: Request):
    """Manually unload a loaded TTS engine. POST with JSON {"model": "omnivoice"}."""
    import json
    body = await req.body()
    try:
        data = json.loads(body)
        name = data.get('model', '')
    except Exception:
        return Response(
            'Bad request. Send {"model": "<engine>"}',
            status_code=400,
            media_type='text/plain',
        )

    if not name:
        # Unload all TTS engines
        freed = await _gc.unload_all()
        return {'unloaded': True, 'models_freed': freed}

    ok = await _gc.unload(name)
    return {'unloaded': ok, 'model': name}


@router.get('/engines')
async def list_engines():
    """List available TTS engines."""
    gc_status = await _gc.status()
    return {
        'available': available_engines(),
        'status': gc_status,
    }
