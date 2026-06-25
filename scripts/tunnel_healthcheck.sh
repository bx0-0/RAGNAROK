#!/bin/bash
#
# Health checks: server readiness + model warmup + model status.
# Requires: PORT, MODEL_NAME from env.
#

source "$(dirname "$(readlink -f "$0")")/tunnel_common.sh"

check_server_ready() {
    local port="${PORT:-8000}"
    echo -ne "  ├─ Waiting for server on port $port"
    for i in $(seq 1 30); do
        sleep 2
        printf "\033[0;36m.\033[0m"
        if curl -s -o /dev/null -w "%{http_code}" \
               "http://localhost:$port/v1/models" 2>/dev/null | grep -qE "200|429"; then
            echo " ✅"
            return 0
        fi
    done
    echo ""
    echo "  ❌ Server not responding on port $port after 60s"
    echo "  Check: curl http://localhost:$port/v1/models"
    exit 1
}

wait_model_warmup() {
    local port="${PORT:-8000}"
    echo -ne "  ├─ Waiting for model warmup"
    for i in $(seq 1 180); do
        STATUS=$(_get_health_status)
        if [ "$STATUS" = "ready" ]; then
            echo " ✅"
            return 0
        fi
        sleep 2
        printf "\033[0;36m.\033[0m"
    done
    echo ""
    echo "  ⚠️  Model warmup didn't finish in time — proceeding anyway"
    return 1
}

show_model_status() {
    echo -ne "  ├─ Checking model status"
    OLLAMA_PS=$(ollama ps 2>/dev/null)
    if [ -n "$OLLAMA_PS" ]; then
        echo " ✅"
        echo ""
        echo -e "  \033[0;36m\033[1m┌── Loaded Model\033[0m"
        HEADR=$(echo "$OLLAMA_PS" | head -1)
        echo -e "  \033[0;36m\033[1m│\033[0m \033[2m${HEADR}\033[0m"
        BODY=$(echo "$OLLAMA_PS" | tail -n +2)
        echo -e "  \033[0;36m\033[1m│\033[0m \033[0;32m${BODY}\033[0m"
        echo -e "  \033[0;36m\033[1m└──────────────────\033[0m"
        echo ""
    else
        echo " ⚠️"
    fi
}
