#!/bin/bash
#
# Background watchdog: monitors tunnel process, restarts if it dies.
# Checks backend health before restarting.
# Requires: PORT, URL_FILE, TUNNEL_LOG_FILE env vars.
#
# Usage: run_tunnel_watchdog <tunnel_pid>
#

source "$(dirname "$(readlink -f "$0")")/tunnel_common.sh"

# ── Print the RAGNAROK online banner ──
print_ragnarok_banner() {
    local url="$1"
    local model_names="$2"
    local port="${PORT:-8000}"
    local first_model=$(echo "$model_names" | awk '{print $1}')

    local display_models=""
    for M in ${MODEL_NAME:-qwen3:8b}; do
        if [[ "$M" == hf.co/* ]]; then
            SHORT=$(echo "$M" | sed 's|hf\.co/[^/]*/||')
            display_models="$display_models $SHORT"
        else
            display_models="$display_models $M"
        fi
    done

    local sep
    sep=$(printf '=%.0s' $(seq 1 56))

    echo ""
    echo -e "\033[0;35m\033[1m  ${sep}\033[0m"
    echo -e "\033[0;32m\033[1m        🔥  RAGNAROK IS ONLINE  🔥         \033[0m"
    echo -e "\033[0;35m\033[1m  ${sep}\033[0m"
    echo -e "\033[0;36m\033[1m  Endpoint\033[2m  \033[1;33m${url}/v1\033[0m"
    echo -e "\033[0;36m\033[1m  Models\033[2m    \033[0;32m${display_models}\033[0m"
    echo -e "\033[0;36m\033[1m  Default\033[2m   \033[0;32m${first_model:-qwen3:8b}\033[0m"
    echo -e "\033[0;36m\033[1m  Port\033[2m      \033[1;37m${port}\033[0m"

    local hf_found=0
    for M in $MODEL_NAME; do
        if [[ "$M" == hf.co/* ]]; then
            hf_found=1
            break
        fi
    done
    if [ "$hf_found" -eq 1 ]; then
        echo ""
        echo -e "  \033[1;33mℹ️  HF models use short aliases. Use the 'Models' name above, not hf.co/...\033[0m"
    fi
    echo -e "\033[0;35m\033[1m  ${sep}\033[0m"
    echo -e "\033[2m  curl \033[1;33m${url}/v1/models\033[2m\033[0m"
    echo -e "\033[0;35m\033[1m  ${sep}\033[0m"
    echo ""
}

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
                if curl -s -o /dev/null -w "%{http_code}" \
                       "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "200"; then
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

            # ── Restart via shared helper ──
            _exec_cloudflared "$port"
            tunnel_pid=$!
            sleep 4

            if kill -0 "$tunnel_pid" 2>/dev/null; then
                local new_url=""
                for _ in $(seq 1 15); do
                    new_url=$(_extract_url_from_logs)
                    if [ -n "$new_url" ]; then break; fi
                    sleep 2
                done

                if [ -n "$new_url" ]; then
                    _write_url_file "$new_url"
                    print_ragnarok_banner "$new_url"
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
