#!/bin/bash
#
# Cloudflare Tunnel manager with auto-restart watchdog
#

set -e

echo "  ├─ Killing old tunnels..."
pkill -f cloudflared 2>/dev/null || true
sleep 1

PUBLIC_URL=""
FAIL_COUNT=0

start_tunnel() {
    echo "  ├─ Starting Cloudflare tunnel..."
    ./cloudflared tunnel --url "http://localhost:${PORT:-8000}" 2>&1 | while IFS= read -r line; do
        if echo "$line" | grep -q "trycloudflare.com"; then
            URL=$(echo "$line" | grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com')
            if [ -n "$URL" ]; then
                echo "$URL" > /tmp/kaggle-ollama-url.txt
                echo ""
                echo "  ✅ Tunnel active!"
                echo "  └─ $URL"
                return 0
            fi
        fi
    done
    return 1
}

echo "  ├─ Waiting for tunnel URL (up to 60s)..."
./cloudflared tunnel --url "http://localhost:${PORT:-8000}" &
TUNNEL_PID=$!

TIMEOUT=60
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if [ -f /tmp/kaggle-ollama-url.txt ]; then
        PUBLIC_URL=$(cat /tmp/kaggle-ollama-url.txt)
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ -n "$PUBLIC_URL" ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║         🎉  KAGGLE OLLAMA GATEWAY IS READY  🎉       ║"
    echo "╠═══════════════════════════════════════════════════════╣"
    echo "║  Public API:                                       ║"
    echo "║  $PUBLIC_URL/v1"
    echo "║  Model: $MODEL_NAME"
    echo "║                                                   ║"
    echo "║  Example:                                          ║"
    echo "║  curl $PUBLIC_URL/v1/chat/completions \\"
    echo "║    -H 'Content-Type: application/json' \\"
    echo "║    -d '{'model':'$MODEL_NAME','messages':..."
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""

    # Watchdog
    while true; do
        if ! kill -0 $TUNNEL_PID 2>/dev/null; then
            echo "⚠️  Tunnel died, restarting in 10s..."
            sleep 10
            ./cloudflared tunnel --url "http://localhost:${PORT:-8000}" &
            TUNNEL_PID=$!
            sleep 5
        fi
        sleep 10
    done
else
    echo "❌ Failed to get tunnel URL within ${TIMEOUT}s"
    echo "   Check Kaggle internet: ping -c 1 trycloudflare.com"
    kill $TUNNEL_PID 2>/dev/null || true
    exit 1
fi
