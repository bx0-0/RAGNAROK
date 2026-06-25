#!/bin/bash
#
# tunnel_common.sh — Shared helpers for all tunnel scripts.
# Source this instead of duplicating cloudflared launch / URL extraction logic.
#
# Required env vars (set by config/settings.env or start.sh):
#   PORT, TUNNEL_LOG_FILE, URL_FILE
#

# ── Launch a new cloudflared tunnel process ──
# Returns the PID on stdout. Handles 429 rate-limit backoff automatically.
_launch_cloudflared() {
    local port="${PORT:-8000}"
    _exec_cloudflared "$port"
    local pid=$!
    sleep 4

    if ! kill -0 "$pid" 2>/dev/null; then
        local log_content
        log_content=$(cat "${TUNNEL_LOG_FILE}" 2>/dev/null || true)
        if echo "$log_content" | grep -q "429"; then
            echo "" >&2
            echo "  ⚠️  Cloudflare rate limited (429). Waiting 30s before retry..." >&2
            pkill -9 -f cloudflared 2>/dev/null || true
            sleep 30
            _launch_cloudflared  # recurse
            return
        else
            echo "" >&2
            echo "  ❌ cloudflared failed to start" >&2
            echo "$log_content" >&2
            exit 1
        fi
    fi
    echo "$pid"
}

# ── Raw exec (no wait, no retry) — for watchdog inline restarts ──
_exec_cloudflared() {
    local port="${1:-${PORT:-8000}}"
    ./cloudflared tunnel \
        --url "http://localhost:${port}" \
        --metrics 0.0.0.0:8282 \
        --no-autoupdate \
        > "${TUNNEL_LOG_FILE}" 2>&1 &
}

# ── Wait for the tunnel URL to appear in logs ──
# Prints URL on stdout; returns 1 on timeout.
_wait_for_tunnel_url() {
    local timeout="${1:-60}"
    local elapsed=0

    while [ "$elapsed" -lt "$timeout" ]; do
        local found_url
        found_url=_extract_url_from_logs
        if [ -n "$found_url" ]; then
            _write_url_file "$found_url"
            echo "$found_url"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        printf "\033[0;36m.\033[0m" >&2
    done
    return 1
}

# ── Extract URL from cloudflared log file ──
_extract_url_from_logs() {
    grep -oP 'https://[a-zA-Z0-9_\-]+\.trycloudflare\.com' \
        "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -1 || true
}

# ── Write URL to file ──
_write_url_file() {
    echo "$1" > "${URL_FILE}"
}

# ── Verify the tunnel endpoint responds ──
_verify_tunnel_endpoint() {
    local url="$1"
    local max_attempts="${2:-30}"

    for _ in $(seq 1 "$max_attempts"); do
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
                     "${url}/v1/models" 2>/dev/null || true)
        if [ "$http_code" = "200" ] || [ "$http_code" = "429" ]; then
            return 0
        fi
        sleep 2
        printf "\033[0;36m.\033[0m" >&2
    done
    return 1
}

# ── Extract health status from /health endpoint (robust) ──
# Uses python3 one-liner instead of fragile grep/cut.
_get_health_status() {
    local port="${PORT:-8000}"
    python3 -c "
import sys, json
try:
    data = open('/dev/stdin').read()
    status = json.loads(data).get('status', '')
    print(status)
except Exception:
    print('')
" < <(curl -s "http://localhost:${port}/health" 2>/dev/null || true)
}
