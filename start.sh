#!/bin/bash
#
# Kaggle Ollama Gateway — Main Launcher
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/settings.env"

MODEL_NAME="qwen3:8b"
MAX_CONCURRENT=1
NUM_CTX=68768
NUM_PREDICT=16384
NUM_BATCH=2444
FLASH_ATTN=True
NUM_GPU=-1
KEEP_ALIVE="60m"
PORT=8000
DEBUG_MODE=False
VERBOSE_LOG=False

if [ -f "$CONFIG_FILE" ]; then
    echo -e "${CYAN}📄 Loading config from $CONFIG_FILE${NC}"
    source "$CONFIG_FILE"
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL_NAME="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --num-ctx)        NUM_CTX="$2"; shift 2 ;;
        --num-predict)    NUM_PREDICT="$2"; shift 2 ;;
        --num-batch)      NUM_BATCH="$2"; shift 2 ;;
        --flash-attn)     FLASH_ATTN="$2"; shift 2 ;;
        --num-gpu)        NUM_GPU="$2"; shift 2 ;;
        --keep-alive)     KEEP_ALIVE="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --debug)          DEBUG_MODE=True; shift ;;
        --verbose-log)    VERBOSE_LOG="$2"; shift 2 ;;
        --help)
            echo "Usage: bash start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>        Ollama model (default: qwen3:8b)"
            echo "  --max-concurrent <n>  Max simultaneous requests (default: 1)"
            echo "  --num-ctx <n>         Context window (default: 68768)"
            echo "  --num-predict <n>     Max output tokens (default: 16384)"
            echo "  --num-batch <n>       Batch size (default: 2444)"
            echo "  --flash-attn <bool>   Flash attention (default: True)"
            echo "  --num-gpu <n>         GPU assignment (default: -1)"
            echo "  --keep-alive <dur>    Keep model loaded (default: 60m)"
            echo "  --port <n>            Server port (default: 8000)"
            echo "  --debug               Enable verbose logging"
            echo "  --verbose-log <bool>    Show request log in terminal (default: True)"
            echo "  --help"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

export MODEL_NAME MAX_CONCURRENT NUM_CTX NUM_PREDICT NUM_BATCH
export FLASH_ATTN NUM_GPU KEEP_ALIVE PORT DEBUG_MODE VERBOSE_LOG

clear
echo ""
echo -e "${BOLD}${GREEN}  ██╗      ██████╗  █████╗ ██████╗ ██╗   ██╗███████╗██████╗ ${NC}"
echo -e "${BOLD}${GREEN}  ██║     ██╔═══██╗██╔══██╗██╔══██╗██║   ██║██╔════╝██╔══██╗${NC}"
echo -e "${BOLD}${GREEN}  ██║     ██║   ██║███████║██████╔╝██║   ██║█████╗  ██████╔╝${NC}"
echo -e "${BOLD}${GREEN}  ██║     ██║   ██║██╔══██║██╔═══╝ ██║   ██║██╔══╝  ██╔══██╗${NC}"
echo -e "${BOLD}${GREEN}  ███████╗╚██████╔╝██║  ██║██║     ╚██████╔╝███████╗██║  ██║${NC}"
echo -e "${BOLD}${GREEN}  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝${NC}"
echo -e "${BOLD}${CYAN}                    Gateway${NC}"
echo ""
echo -e "  ${YELLOW}Model:${NC}      ${BOLD}$MODEL_NAME${NC}"
echo -e "  ${YELLOW}Concurrent:${NC} ${BOLD}$MAX_CONCURRENT${NC}"
echo -e "  ${YELLOW}Context:${NC}    ${BOLD}$NUM_CTX${NC}"
echo -e "  ${YELLOW}Port:${NC}       ${BOLD}$PORT${NC}"
echo ""

# Step 1
echo -e "${YELLOW}[1/4]${NC} Installing dependencies..."
bash "$SCRIPT_DIR/scripts/setup.sh"

# Step 2
echo -e "${YELLOW}[2/4]${NC} Preparing Ollama & model..."
bash "$SCRIPT_DIR/scripts/install_model.sh"

# Step 3
echo -e "${YELLOW}[3/4]${NC} Starting FastAPI server..."
cd "$SCRIPT_DIR"
pkill -f "src.server" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

python3 -m src.server &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Enable verbose print if requested
if [ "$VERBOSE_LOG" = "True" ] || [ "$VERBOSE_LOG" = "true" ]; then
    echo "  [verbose-log enabled — request log will appear below after tunnel setup]"
fi

echo -n "  Waiting for server"
READY=0
for i in $(seq 1 30); do
    sleep 2
    printf "."
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
        READY=1
        break
    fi
done
if [ "$READY" -ne 1 ]; then
    echo ""
    echo -e "  ${RED}❌ Server failed to start${NC}"
    tail -20 /tmp/gateway-server.log 2>/dev/null
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}✅${NC}"

# Step 4
echo -e "${YELLOW}[4/4]${NC} Creating Cloudflare tunnel..."
bash "$SCRIPT_DIR/scripts/tunnel.sh"
