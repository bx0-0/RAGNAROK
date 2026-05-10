#!/bin/bash
#
# Cloudflare Tunnel manager with auto-restart watchdog
#

set -e

echo "  ├─ Killing old tunnels..."
pkill -f cloudflared 2>/dev/null || true
sleep 1

PORT="${PORT:-8000}"
URL_FILE="/tmp/kaggle-ollama-url.txt"
rm -f "$URL_FILE"

# ---- Health check: wait for server ----
echo "  ├─ Waiting for server on port $PORT..."
SERVER_READY=0
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qE "200|429"; then
        SERVER_READY=1
        break
    fi
    sleep 2
done

if [ "$SERVER_READY" -ne 1 ]; then
    echo "  ❌ Server not responding on port $PORT after 60s"
    echo "  Check: curl http://localhost:$PORT/v1/models"
    exit 1
fi
echo "  ✅ Server is responding"

# ---- Start tunnel and capture URL ----
echo "  ├─ Starting Cloudflare tunnel..."

./cloudflared tunnel --url "http://localhost:${PORT}" > /tmp/cloudflared.log 2>&1 &
TUNNEL_PID=$!
sleep 2

if ! kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "  ❌ cloudflared failed to start"
    cat /tmp/cloudflared.log
    exit 1
fi

# Wait for URL in logs
TIMEOUT=60
ELAPSED=0
PUBLIC_URL=""

while [ $ELAPSED -lt $TIMEOUT ]; do
    if [ -f /tmp/cloudflared.log ]; then
        FOUND_URL=$(grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | tail -1)
        if [ -n "$FOUND_URL" ]; then
            PUBLIC_URL="$FOUND_URL"
            echo "$FOUND_URL" > "$URL_FILE"
            break
        fi
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ -n "$PUBLIC_URL" ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           🎉  KAGGLE OLLAMA GATEWAY IS READY  🎉         ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║  Public API:                                             ║"
    echo "║  ${PUBLIC_URL}/v1                                       ║"
    echo "║  Model: ${MODEL_NAME:-qwen3:8b}                                       ║"
    echo "║                                                          ║"
    echo "║  Test it:                                                ║"
    echo "║  curl ${PUBLIC_URL}/v1/models                              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Watchdog in background
    (
        while true; do
            if ! kill -0 $TUNNEL_PID 2>/dev/null; then
                echo "⚠️  Tunnel died, restarting in 10s..."
                sleep 10
                ./cloudflared tunnel --url "http://localhost:${PORT}" > /tmp/cloudflared.log 2>&1 &
                TUNNEL_PID=$!
                sleep 5
            fi
            sleep 10
        done
    ) &

    # If verbose-log enabled, tail the request log live in the cell
    if [ "${VERBOSE_LOG}" = "True" ] || [ "${VERBOSE_LOG}" = "true" ]; then
        echo ""
        echo -e "${GREEN}─── Request Log (live) ───${NC}"
        tail -f /tmp/gateway-requests.log
    else
        # Keep script alive silently
        while true; do sleep 60; done
    fi
else
    echo "❌ Failed to get tunnel URL within ${TIMEOUT}s"
    echo "   Log output:"
    cat /tmp/cloudflared.log 2>/dev/null | tail -20
    kill $TUNNEL_PID 2>/dev/null || true
    exit 1
fi
