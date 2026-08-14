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
from src.tts import get_engine_class, available_engines
from src.logging import log_request_start, log_request, logger
from src.config import TTS_ENABLED, TTS_MAX_CHARS, TTS_MIN_GPU_FREE_GB

router = APIRouter(prefix='/v1/audio')
_gc = None  # set lazily — ModelGC singleton is created in server lifespan


def _get_gc():
    global _gc
    if _gc is None:
        _gc = ModelGC.get()
    return _gc


async def _get_or_load_engine(name: str):
    """Return an engine instance, loading it lazily if needed."""
    cls = get_engine_class(name)
    eng = cls.get()

    # Register with GC on first use
    gc = _get_gc()
    await gc.register(
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

    # Log request start for verbose log
    req_id = req.headers.get('x-request-id', req.headers.get('x-requested-with', 'tts'))
    await log_request_start(req_id, 'POST', '/v1/audio/speech',
                             extra=f'model={request.model}, chars={len(request.input)}')
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

    # ── GPU memory guard for GPU engines ─────────────────────────────
    if eng.device == 'cuda':
        try:
            import torch
        except Exception:
            logger.warning('torch not available — skipping GPU memory check')
        else:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                reserved = torch.cuda.memory_reserved(0)
                allocated = torch.cuda.memory_allocated(0)
                free = total - reserved
                free_gb = free / (1024 ** 3)
                min_gb = TTS_MIN_GPU_FREE_GB
                if free_gb < min_gb:
                    logger.warning(
                        f'GPU memory low: {free_gb:.2f} GB free < {min_gb} GB. Unloading LLMs...'
                    )
                    # Free GPU by unloading idle LLMs
                    gc = _get_gc()
                    # Try unload all (or specific LLM unloading logic)
                    await gc.unload_all()  # or call specific LLM unload
                    # Re-check after unload
                    reserved2 = torch.cuda.memory_reserved(0)
                    free2 = total - reserved2
                    free_gb2 = free2 / (1024 ** 3)
                    if free_gb2 < min_gb:
                        return Response(
                            content=f'GPU memory insufficient ({free_gb2:.2f} GB free < {min_gb} GB). '
                                   f'Unload large models or increase VRAM.',
                            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
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
    _get_gc().touch(engine_name)

    elapsed = round(time.monotonic() - t0, 3)

    await log_request(
        req_id, 'POST', '/v1/audio/speech',
        HTTPStatus.OK, elapsed, int(t0), int(time.monotonic()),
        extra=f'{engine_name} {len(audio_bytes):,}B',
    )

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
        freed = await _get_gc().unload_all()
        return {'unloaded': True, 'models_freed': freed}

    ok = await _get_gc().unload(name)
    return {'unloaded': ok, 'model': name}


@router.get('/engines')
async def list_engines():
    """List available TTS engines."""
    gc_status = await _get_gc().status()
    return {
        'available': available_engines(),
        'status': gc_status,
    }
