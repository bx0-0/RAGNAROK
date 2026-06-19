#!/bin/bash
#
# Background watchdog: monitors tunnel process, restarts if it dies.
# Checks backend health before restarting.
# Requires: PORT, URL_FILE, TUNNEL_LOG_FILE env vars
#
# Usage: run_tunnel_watchdog <tunnel_pid>
#

source "$(dirname "$(readlink -f "$0")")/tunnel_url_extract.sh"

run_tunnel_watchdog() {
    local tunnel_pid="$1"
    local port="${PORT:-8000}"
    local restart_count=0

    while true; do
        if ! kill -0 "$tunnel_pid" 2>/dev/null; then
            restart_count=$((restart_count + 1))
            echo "⚠️  Tunnel died (restart #${restart_count}), checking backend..."

            # ── Check backend is alive ──
            local backend_up=0
            for _ in $(seq 1 5); do
                if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "200"; then
                    backend_up=1
                    break
                fi
                sleep 1
            done

            if [ "$backend_up" -eq 0 ]; then
                echo "  ❌ Backend on port ${port} is DOWN. Not restarting tunnel."
                echo "  Check: curl http://localhost:${port}/health"
                cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -10
                exit 1
            fi

            echo "  └─ Backend OK, dumping tunnel logs..."
            cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -5

            # ── Handle 429 backoff ──
            if grep -q "429" "${TUNNEL_LOG_FILE}" 2>/dev/null; then
                echo "  ⚠️  Cloudflare rate limited, waiting 30s..."
                sleep 30
            fi

            sleep 5
            pkill -9 -f cloudflared 2>/dev/null || true
            sleep 2

            # ── Restart tunnel ──
            ./cloudflared tunnel --url "http://localhost:${port}" --metrics 0.0.0.0:8282 --no-autoupdate > "${TUNNEL_LOG_FILE}" 2>&1 &
            tunnel_pid=$!
            sleep 4

            if kill -0 "$tunnel_pid" 2>/dev/null; then
                # Extract new URL
                local new_url=""
                for _ in $(seq 1 15); do
                    new_url=$(extract_tunnel_url || true)
                    if [ -n "$new_url" ]; then break; fi
                    sleep 2
                done

                if [ -n "$new_url" ]; then
                    write_url_file "$new_url"
                    echo "  ✅ Tunnel restarted — NEW URL: ${new_url}/v1"
                else
                    echo "  ✅ Tunnel restarted (URL extraction failed, check logs)"
                fi
            else
                echo "  ❌ Tunnel restart failed"
                cat "${TUNNEL_LOG_FILE}" | tail -10
                exit 1
            fi
        fi
        sleep 15
    done
}