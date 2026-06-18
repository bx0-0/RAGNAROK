"""Logging setup + helpers shared across all modules."""

import os
import time
import collections
import logging

logger = logging.getLogger("gateway")

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "True").lower() in ("true", "1", "yes")
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/tmp/gateway-requests.log")


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


# ─── Async-safe log file (non-blocking, line-buffered for tail -f) ───
_log_fh = None
_request_log = collections.deque(maxlen=500)


def _open_log_fh():
    global _log_fh
    if _log_fh is None:
        _log_fh = open(REQUEST_LOG_FILE, "a", buffering=1)


def _status_color(code):
    if code < 300:
        return "\033[0;32m"
    if code == 429:
        return "\033[0;33m"
    if code < 500:
        return "\033[0;31m"
    return "\033[0;35m"


def _status_label(code):
    if code < 300:
        return "OK"
    if code == 429:
        return "Busy"
    if code == 400:
        return "Bad"
    if code < 500:
        return "Err"
    return "Fatal"


def _build_log_line(tag, req_id, status_or_method, path=None, extra=None, duration=None, t_in=None, t_out=None):
    ts = time.strftime("%H:%M:%S")
    line = f"\033[0;36m[{ts}]\033[0m \033[0;90m{tag}\033[0m \033[1m{req_id}\033[0m "
    if duration is not None:
        color = _status_color(status_or_method)
        label = _status_label(status_or_method)
        line += (
            f"{color}{status_or_method} {label}\033[0m "
            f"{duration}s "
            f"\033[0;90mt:{t_in}\u2192{t_out}\033[0m"
        )
    else:
        line += f"{status_or_method} {path}"
    if extra:
        line += f"  \033[0;33m{extra}\033[0m"
    return line


def _enqueue_log(line: str):
    _open_log_fh()
    try:
        _log_fh.write(line + "\n")
    except OSError:
        pass


async def log_request_start(req_id, method, path, extra=""):
    line = _build_log_line("◀", req_id, method, path, extra=extra)
    if VERBOSE_LOG:
        print(line, flush=True)
    _enqueue_log(line)


async def log_request(req_id, method, path, status, duration, t_in, t_out, extra=""):
    _request_log.append({
        "id": req_id, "method": method, "path": path,
        "status": status, "duration": duration,
        "t_in": t_in, "t_out": t_out, "extra": extra,
    })
    line = _build_log_line(
        "▶", req_id, status, path,
        duration=duration, t_in=t_in, t_out=t_out, extra=extra,
    )
    if VERBOSE_LOG:
        print(line, flush=True)
    _enqueue_log(line)
