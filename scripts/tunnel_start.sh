#!/bin/bash
#
# Start Cloudflare tunnel with retry logic for 429 rate limits
# Requires: PORT, URL_FILE, TUNNEL_LOG_FILE env vars
#

# Source URL extraction helpers
source "$(dirname "$(readlink -f "$0")")/tunnel_url_extract.sh"

# ── Recursive start with 429 backoff ──
start_tunnel() {
    local port="${PORT:-8000}"
    ./cloudflared tunnel --url "http://localhost:${port}" --metrics 0.0.0.0:8282 --no-autoupdate > "${TUNNEL_LOG_FILE}" 2>&1 &
    local pid=$!
    sleep 4

    if ! kill -0 $pid 2>/dev/null; then
        local log_content=$(cat "${TUNNEL_LOG_FILE}" 2>/dev/null)
        if echo "$log_content" | grep -q "429"; then
            echo ""
            echo "  ⚠️  Cloudflare rate limited (429). Waiting 30s before retry..."
            pkill -9 -f cloudflared 2>/dev/null || true
            sleep 30
            start_tunnel
        else
            echo ""
            echo "  ❌ cloudflared failed to start"
            echo "$log_content"
            exit 1
        fi
    fi
    echo $pid
}

# ── Wait for URL to appear in tunnel logs ──
wait_for_tunnel_url() {
    local timeout="${1:-60}"
    local elapsed=0
    local found_url=""

    while [ $elapsed -lt $timeout ]; do
        found_url=$(extract_tunnel_url)
        if [ -n "$found_url" ]; then
            write_url_file "$found_url"
            echo "$found_url"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        printf "\033[0;36m.\033[0m" >&2
    done
    return 1
}

# ── Verify tunnel endpoint is reachable ──
verify_tunnel_endpoint() {
    local url="$1"
    local max_attempts="${2:-30}"

    for _ in $(seq 1 $max_attempts); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${url}/v1/models" 2>/dev/null || true)
        if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "429" ]; then
            return 0
        fi
        sleep 2
        printf "\033[0;36m.\033[0m" >&2
    done
    return 1
}

# ── Full retry: kill old tunnel, relaunch, extract URL, verify ──
retry_tunnel_full() {
    local port="${PORT:-8000}"
    kill $1 2>/dev/null || true
    pkill -9 -f cloudflared 2>/dev/null || true
    sleep 3

    ./cloudflared tunnel --url "http://localhost:${port}" --metrics 0.0.0.0:8282 --no-autoupdate > "${TUNNEL_LOG_FILE}" 2>&1 &
    local new_pid=$!
    sleep 5

    local new_url=""
    for _ in $(seq 1 15); do
        new_url=$(extract_tunnel_url || true)
        if [ -n "$new_url" ]; then break; fi
        sleep 2
    done

    if [ -z "$new_url" ]; then
        echo "" >&2
        echo "  ❌ Could not get new URL on retry" >&2
        cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -10 >&2
        exit 1
    fi

    write_url_file "$new_url"
    echo -ne "  ├─ Testing retry tunnel" >&2
    if ! verify_tunnel_endpoint "$new_url"; then
        echo "" >&2
        echo "  ❌ Retry also failed. Tunnel may be blocked." >&2
        cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -10 >&2
        kill $new_pid 2>/dev/null || true
        exit 1
    fi
    echo " ✅" >&2

    echo "$new_pid:$new_url"
}