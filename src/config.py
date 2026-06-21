"""All configuration — env vars, defaults, and derived values."""

import os

# ─── Model config ───
_RAW_MODEL_LIST = os.environ.get("MODEL_NAME", "qwen3:8b").split()
_SHORT_ALIASES = {name: name.split("/", 3)[-1] for name in _RAW_MODEL_LIST if name.startswith("hf.co/")}
_MODEL_LIST = [_SHORT_ALIASES.get(m, m) for m in _RAW_MODEL_LIST]
MODEL_NAME = _MODEL_LIST[0]  # Default = first model

# ─── Server config ───
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "16384"))
NUM_BATCH = int(os.environ.get("NUM_BATCH", "2444"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "True").lower() in ("true", "1", "yes")
NUM_GPU = int(os.environ.get("NUM_GPU", "-1"))
KEEP_ALIVE = os.environ.get("KEEP_ALIVE", "60m")
PORT = int(os.environ.get("PORT", "8000"))

# ─── Ollama config ───
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

# ─── Logging config ───
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "True").lower() in ("true", "1", "yes")
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/tmp/gateway-requests.log")

# ─── Ollama stream debug log ───
OLLAMA_STREAM_LOG = os.environ.get("OLLAMA_STREAM_LOG", "/tmp/ollama-stream.log")

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
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "256"))
MAX_KEEPALIVE_PINGS = int(os.environ.get("MAX_KEEPALIVE_PINGS", "120"))

# ─── Precomputed ollama options (never changes at runtime) ───
_OLLAMA_OPTS_BASE = {
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
}

_OLLAMA_OPTS = {
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": NUM_PREDICT,
}

_OLLAMA_OPTS_WARMUP = {
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": 1,
}
