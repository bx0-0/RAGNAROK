#!/bin/bash
#
# Start Cloudflare tunnel with retry logic for 429 rate limits.
# Requires: PORT, URL_FILE, TUNNEL_LOG_FILE env vars.
#

source "$(dirname "$(readlink -f "$0")")/tunnel_common.sh"

# ── Start tunnel (handles 429 backoff via _launch_cloudflared) ──
start_tunnel() {
    echo "$(_launch_cloudflared)"
}

# ── Wait for URL to appear in tunnel logs ──
wait_for_tunnel_url() {
    local timeout="${1:-60}"
    _wait_for_tunnel_url "$timeout"
}

# ── Verify tunnel endpoint is reachable ──
verify_tunnel_endpoint() {
    local url="$1"
    _verify_tunnel_endpoint "$url"
}

# ── Full retry: kill old tunnel, relaunch, extract URL, verify.
#
# FIX: Use tab as delimiter so colons in https:// URLs don't break parsing.
#      Callers do: read -r new_pid new_url <<< "$(retry_tunnel_full $old_pid)"
# ──
retry_tunnel_full() {
    local port="${PORT:-8000}"

    kill "$1" 2>/dev/null || true
    pkill -9 -f cloudflared 2>/dev/null || true
    sleep 3

    _exec_cloudflared "$port"
    local new_pid=$!
    sleep 5

    local new_url=""
    for _ in $(seq 1 15); do
        new_url=$(_extract_url_from_logs)
        if [ -n "$new_url" ]; then break; fi
        sleep 2
    done

    if [ -z "$new_url" ]; then
        echo "" >&2
        echo "  ❌ Could not get new URL on retry" >&2
        cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -10 >&2
        exit 1
    fi

    _write_url_file "$new_url"
    echo -ne "  ├─ Testing retry tunnel" >&2
    if ! _verify_tunnel_endpoint "$new_url"; then
        echo "" >&2
        echo "  ❌ Retry also failed. Tunnel may be blocked." >&2
        cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -10 >&2
        kill "$new_pid" 2>/dev/null || true
        exit 1
    fi
    echo " ✅" >&2

    # Tab delimiter — safe for URLs containing colons
    printf "%s\t%s" "$new_pid" "$new_url"
}
