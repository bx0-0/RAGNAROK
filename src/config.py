"""All configuration — env vars, defaults, and derived values."""

import os

# ─── Model config ───
_RAW_MODEL_LIST = os.environ.get("MODEL_NAME", "qwen3.5:9b").split()
_SHORT_ALIASES = {name: name.split("/", 3)[-1] for name in _RAW_MODEL_LIST if name.startswith("hf.co/")}
_MODEL_LIST = [_SHORT_ALIASES.get(m, m) for m in _RAW_MODEL_LIST]
MODEL_NAME = _MODEL_LIST[0]  # Default = first model

# ─── Server config ───
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
_NUM_CTX_RAW = os.environ.get("NUM_CTX", "16384").split()
if len(_NUM_CTX_RAW) not in (1, len(_MODEL_LIST)):
    raise ValueError(
        f"NUM_CTX accepts 1 value (applied to all models) or exactly "
        f"{len(_MODEL_LIST)} (one per model: {_MODEL_LIST}). "
        f"Got {len(_NUM_CTX_RAW)}.\n"
        f"Example: NUM_CTX='100000 50000 32000'"
    )
_NUM_CTX_VALS = [int(v) for v in _NUM_CTX_RAW]
MODEL_NUM_CTX: dict[str, int] = (
    {m: _NUM_CTX_VALS[0] for m in _MODEL_LIST}
    if len(_NUM_CTX_VALS) == 1
    else dict(zip(_MODEL_LIST, _NUM_CTX_VALS))
)
NUM_CTX = _NUM_CTX_VALS[0]  # Backward-compat: first value (banner display)
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "16384"))
NUM_BATCH = int(os.environ.get("NUM_BATCH", "500"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "True").lower() in ("true", "1", "yes")
NUM_GPU = int(os.environ.get("NUM_GPU", "-1"))
KEEP_ALIVE = os.environ.get("KEEP_ALIVE", "60m")
PORT = int(os.environ.get("PORT", "8000"))

# ─── Ollama config ───
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── TTS config ───
TTS_ENABLED = os.environ.get("TTS_ENABLED", "True").lower() in ("true", "1", "yes")
TTS_DEFAULT_ENGINE = os.environ.get("TTS_DEFAULT_ENGINE", "omnivoice")  # omnivoice | inflect
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "5000"))
TTS_MIN_GPU_FREE_GB = float(os.environ.get("TTS_MIN_GPU_FREE_GB", "7"))
# OmniVoice & Inflect param defaults are in src/models/tts.py — sent per-request

# ─── Reasoning effort mapping ───
# Client value (lowercase) → Ollama `think` value.
# Add new levels here when Ollama supports them — single edit point.
# Unknown values fall back to True (thinking on, model default level).
THINK_LEVEL_MAP = {
    "none": False,
    "off": False,
    "false": False,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}

# ─── GC config ───
GC_IDLE_TIMEOUT = float(os.environ.get("GC_IDLE_TIMEOUT", "600.0"))  # 10 min default
GC_SWEEP_INTERVAL = float(os.environ.get("GC_SWEEP_INTERVAL", "60.0"))  # 1 min default


# ─── Logging config ───
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "True").lower() in ("true", "1", "yes")
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/tmp/gateway-requests.log")

# ─── HTTP timeouts ───
MAX_STREAM_SECONDS = int(os.environ.get("MAX_STREAM_SECONDS", "1800"))  # 30 min
HTTP_CONNECT_TIMEOUT = float(os.environ.get("HTTP_CONNECT_TIMEOUT", "60.0"))
HTTP_READ_TIMEOUT = float(os.environ.get("HTTP_READ_TIMEOUT", "7200.0"))  # 2h for large tool-call outputs
HTTP_WRITE_TIMEOUT = float(os.environ.get("HTTP_WRITE_TIMEOUT", "60.0"))
HTTP_POOL_TIMEOUT = float(os.environ.get("HTTP_POOL_TIMEOUT", "900.0"))

# ─── HTTP connection pool ───
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "2000"))
MAX_KEEPALIVE_CONNECTIONS = int(os.environ.get("MAX_KEEPALIVE_CONNECTIONS", "500"))
KEEPALIVE_EXPIRY = int(os.environ.get("KEEPALIVE_EXPIRY", "300"))

# ─── Per-model Ollama options ───
_OLLAMA_OPTS_BASE = {
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": NUM_PREDICT,
}


def model_opts(model: str, *, warmup: bool = False) -> dict:
    """Return Ollama options for a specific model (per-model num_ctx).

    The only per-model field is ``num_ctx``; all others are shared.
    ``warmup=True`` overrides ``num_predict`` to 1 (quick load check).
    """
    opts = dict(_OLLAMA_OPTS_BASE)
    opts["num_ctx"] = MODEL_NUM_CTX.get(model, NUM_CTX)
    if warmup:
        opts["num_predict"] = 1
    return opts
