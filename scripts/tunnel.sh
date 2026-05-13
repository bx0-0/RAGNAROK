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

GREEN='\033[0;32m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
WHITE='\033[1;37m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---- Health check: wait for server ----
echo -ne "  ├─ Waiting for server on port $PORT"
SERVER_READY=0
for i in $(seq 1 30); do
    sleep 2
    printf "\033[0;36m.\033[0m"
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qE "200|429"; then
        SERVER_READY=1
        break
    fi
done

if [ "$SERVER_READY" -ne 1 ]; then
    echo ""
    echo "  ❌ Server not responding on port $PORT after 60s"
    echo "  Check: curl http://localhost:$PORT/v1/models"
    exit 1
fi
echo " ✅"

# ---- Wait for model warmup ----
echo -ne "  ├─ Waiting for model warmup"
WARM_READY=0
for i in $(seq 1 180); do
    STATUS=$(curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    if [ "$STATUS" = "ready" ]; then
        WARM_READY=1
        break
    fi
    sleep 2
    printf "\033[0;36m.\033[0m"
done

if [ "$WARM_READY" -ne 1 ]; then
    echo ""
    echo "  ⚠️  Model warmup didn't finish in time — proceeding anyway"
else
    echo " ✅"
fi

# ---- Show model status ----
echo -ne "  ├─ Checking model status"
OLLAMA_PS=$(ollama ps 2>/dev/null)
if [ -n "$OLLAMA_PS" ]; then
    echo " ✅"
    echo ""
    echo -e "  ${CYAN}${BOLD}┌── Loaded Model${NC}"
    HEADR=$(echo "$OLLAMA_PS" | head -1)
    echo -e "  ${CYAN}${BOLD}│${NC} ${DIM}${HEADR}${NC}"
    BODY=$(echo "$OLLAMA_PS" | tail -n +2)
    echo -e "  ${CYAN}${BOLD}│${NC} ${GREEN}${BODY}${NC}"
    echo -e "  ${CYAN}${BOLD}└──────────────────${NC}"
    echo ""
else
    echo " ⚠️"
fi

# ---- Start tunnel and capture URL ----
echo -ne "  ├─ Starting Cloudflare tunnel"

./cloudflared tunnel --url "http://localhost:${PORT}" --metrics 0.0.0.0:8282 > /tmp/cloudflared.log 2>&1 &
TUNNEL_PID=$!
sleep 3

if ! kill -0 $TUNNEL_PID 2>/dev/null; then
    echo ""
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
    printf "\033[0;36m.\033[0m"
done

if [ -n "$PUBLIC_URL" ]; then
    echo " ✅"

    # Watchdog in background
    (
        RESTART_COUNT=0
        while true; do
            if ! kill -0 $TUNNEL_PID 2>/dev/null; then
                RESTART_COUNT=$((RESTART_COUNT + 1))
                echo "⚠️  Tunnel died (restart #${RESTART_COUNT}), checking backend..."

                # Check if backend is still alive
                BACKEND_UP=0
                for _b in $(seq 1 5); do
                    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
                        BACKEND_UP=1
                        break
                    fi
                    sleep 1
                done

                if [ "$BACKEND_UP" -eq 0 ]; then
                    echo "  ❌ Backend on port ${PORT} is DOWN. Not restarting tunnel."
                    echo "  Check: curl http://localhost:${PORT}/health"
                    cat /tmp/cloudflared.log 2>/dev/null | tail -10
                    exit 1
                fi

                echo "  └─ Backend OK, dumping tunnel logs..."
                cat /tmp/cloudflared.log 2>/dev/null | tail -5

                sleep 5
                pkill -f cloudflared 2>/dev/null || true
                sleep 2

                ./cloudflared tunnel --url "http://localhost:${PORT}" --metrics 0.0.0.0:8282 > /tmp/cloudflared.log 2>&1 &
                TUNNEL_PID=$!
                sleep 3

                if kill -0 $TUNNEL_PID 2>/dev/null; then
                    echo "  ✅ Tunnel restarted"
                else
                    echo "  ❌ Tunnel restart failed"
                    cat /tmp/cloudflared.log | tail -10
                    exit 1
                fi
            fi
            sleep 15
        done
    ) &

    # ── Banner ──
    SEP=$(printf '=%.0s' $(seq 1 56))
    echo ""
    echo -e "${MAGENTA}${BOLD}  ${SEP}${NC}"
    echo -e "${GREEN}${BOLD}        🔥  RAGNAROK IS ONLINE  🔥         ${NC}"
    echo -e "${MAGENTA}${BOLD}  ${SEP}${NC}"
    echo -e "${CYAN}${BOLD}  Endpoint${DIM}  ${YELLOW}${PUBLIC_URL}/v1${NC}"
    echo -e "${CYAN}${BOLD}  Model${DIM}     ${GREEN}${MODEL_NAME:-qwen3:8b}${NC}"
    echo -e "${CYAN}${BOLD}  Port${DIM}      ${WHITE}${PORT:-8000}${NC}"
    echo -e "${MAGENTA}${BOLD}  ${SEP}${NC}"
    echo -e "${DIM}  curl ${YELLOW}${PUBLIC_URL}/v1/models${DIM}${NC}"
    echo -e "${MAGENTA}${BOLD}  ${SEP}${NC}"
    echo ""

    # If verbose-log enabled, tail the request log live in the cell
    if [ "${VERBOSE_LOG}" = "True" ] || [ "${VERBOSE_LOG}" = "true" ]; then
        echo ""
        echo -e "${GREEN}${BOLD}  ═══ Request Log (live) ═══${NC}"
        for _wait in $(seq 1 15); do
            if [ -f /tmp/gateway-requests.log ]; then break; fi
            sleep 1
        done
        tail -f /tmp/gateway-requests.log
    else
        while true; do sleep 60; done
    fi
else
    echo ""
    echo "❌ Failed to get tunnel URL within ${TIMEOUT}s"
    echo "   Log output:"
    cat /tmp/cloudflared.log 2>/dev/null | tail -20
    kill $TUNNEL_PID 2>/dev/null || true
    exit 1
fi
