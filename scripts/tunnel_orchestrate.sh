#!/bin/bash
#
# Tunnel orchestrator — replaces the old monolithic tunnel.sh.
# Sources sub-scripts: healthcheck, start, watchdog.
#

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Source config if not already loaded
if [ -z "$URL_FILE" ]; then
    source "$(dirname "$SCRIPT_DIR")/config/settings.env"
    export URL_FILE TUNNEL_LOG_FILE REQUEST_LOG_FILE
fi

# Source sub-scripts (each sources tunnel_common.sh internally)
source "$SCRIPT_DIR/tunnel_healthcheck.sh"
source "$SCRIPT_DIR/tunnel_start.sh"
source "$SCRIPT_DIR/tunnel_watchdog.sh"

# ── Cleanup old tunnels ──
echo "  ├─ Killing old tunnels..."
pkill -f cloudflared 2>/dev/null || true
sleep 1

rm -f "$URL_FILE"

# ── Phase 1: Health checks ──
check_server_ready
wait_model_warmup
show_model_status

# ── Phase 2: Kill any remaining cloudflared instances ──
pkill -9 -f cloudflared 2>/dev/null || true
sleep 3

# ── Phase 3: Start tunnel + get URL ──
echo -ne "  ├─ Starting Cloudflare tunnel"
TUNNEL_PID=$(start_tunnel)

PUBLIC_URL=$(wait_for_tunnel_url 60)
if [ -z "$PUBLIC_URL" ]; then
    echo ""
    echo "❌ Failed to get tunnel URL within 60s"
    echo "   Log output:"
    cat "${TUNNEL_LOG_FILE}" 2>/dev/null | tail -20
    kill $TUNNEL_PID 2>/dev/null || true
    exit 1
fi
echo " ✅"

# ── Phase 4: Verify tunnel endpoint ──
# Wait briefly for DNS propagation before first check
sleep 2
if ! verify_tunnel_endpoint "$PUBLIC_URL"; then
    echo ""
    echo "  ├─ Initial tunnel not ready, retrying..."

    # FIX: Use tab delimiter instead of colon so https:// doesn't break parsing.
    read -r new_pid new_url <<< "$(retry_tunnel_full "$TUNNEL_PID")"
    TUNNEL_PID="$new_pid"
    PUBLIC_URL="$new_url"
else
    echo " ✅"
fi

# ── Phase 5: Launch watchdog in background ──
run_tunnel_watchdog "$TUNNEL_PID" &

# ── Phase 6: Banner ──
SEP=$(printf '=%.0s' $(seq 1 56))

# Convert HF names to display-friendly short aliases
DISPLAY_MODEL_NAMES=""
for M in ${MODEL_NAME:-qwen3.5:9b}; do
    if [[ "$M" == hf.co/* ]]; then
        SHORT=$(echo "$M" | sed 's|hf\.co/[^/]*/||')
        DISPLAY_MODEL_NAMES="$DISPLAY_MODEL_NAMES $SHORT"
    else
        DISPLAY_MODEL_NAMES="$DISPLAY_MODEL_NAMES $M"
    fi
done

echo ""
echo -e "\033[0;35m\033[1m  ${SEP}\033[0m"
echo -e "\033[0;32m\033[1m        🔥  RAGNAROK IS ONLINE  🔥         \033[0m"
echo -e "\033[0;35m\033[1m  ${SEP}\033[0m"
echo -e "\033[0;36m\033[1m  Endpoint\033[2m  \033[1;33m${PUBLIC_URL}/v1\033[0m"
echo -e "\033[0;36m\033[1m  Models\033[2m    \033[0;32m${DISPLAY_MODEL_NAMES}\033[0m"
FIRST_SHORT=$(echo "$DISPLAY_MODEL_NAMES" | awk '{print $1}')
echo -e "\033[0;36m\033[1m  Default\033[2m   \033[0;32m${FIRST_SHORT:-qwen3.5:9b}\033[0m"
echo -e "\033[0;36m\033[1m  Port\033[2m      \033[1;37m${PORT:-8000}\033[0m"

# Hint if any HF model was used
HF_MODEL_FOUND=0
for M in $MODEL_NAME; do
    if [[ "$M" == hf.co/* ]]; then
        HF_MODEL_FOUND=1
        break
    fi
done
if [ "$HF_MODEL_FOUND" -eq 1 ]; then
    echo ""
    echo -e "  \033[1;33mℹ️  HF models use short aliases. Use the 'Models' name above, not hf.co/...\033[0m"
fi
echo -e "\033[0;35m\033[1m  ${SEP}\033[0m"
echo -e "\033[2m  curl \033[1;33m${PUBLIC_URL}/v1/models\033[2m\033[0m"
echo -e "\033[0;35m\033[1m  ${SEP}\033[0m"
echo ""
echo -e "  \033[2mCreated by \033[1;37mSaber Mohamed\033[0;2m  |  RAGNAROK Gateway\033[0m"
echo ""

# ── Phase 7: Runtime loop ──
if [ "${VERBOSE_LOG}" = "True" ] || [ "${VERBOSE_LOG}" = "true" ]; then
    echo ""
    echo -e "\033[0;32m\033[1m  ═══ Request Log (live) ═══\033[0m"
    for _wait in $(seq 1 15); do
        if [ -f "$REQUEST_LOG_FILE" ]; then break; fi
        sleep 1
    done
    tail -f "$REQUEST_LOG_FILE"
else
    while true; do sleep 60; done
fi
