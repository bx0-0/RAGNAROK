#!/bin/bash
#
# Kaggle Ollama Gateway — Main Launcher
#
# Usage:
#   bash start.sh                          # Use config/settings.env
#   bash start.sh --model qwen3:8b         # Override model
#   bash start.sh --model qwen3:8b --max-concurrent 2
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Resolve project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/settings.env"

# ---- Default values ----
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

# ---- Load config file if exists ----
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${CYAN}📄 Loading config from $CONFIG_FILE${NC}"
    source "$CONFIG_FILE"
fi

# ---- Parse CLI args (override config) ----
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
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ---- Export all config ----
export MODEL_NAME MAX_CONCURRENT NUM_CTX NUM_PREDICT NUM_BATCH
export FLASH_ATTN NUM_GPU KEEP_ALIVE PORT DEBUG_MODE

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Kaggle Ollama Gateway — Starting       ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Model:      ${CYAN}$MODEL_NAME${NC}                  ${GREEN}║${NC}"
echo -e "${GREEN}║  Concurrent: ${CYAN}$MAX_CONCURRENT${NC}                   ${GREEN}║${NC}"
echo -e "${GREEN}║  Context:    ${CYAN}$NUM_CTX${NC}                    ${GREEN}║${NC}"
echo -e "${GREEN}║  Port:       ${CYAN}$PORT${NC}                     ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ---- Step 1: Setup ----
echo -e "${YELLOW}⚙️  Step 1/4: Installing dependencies...${NC}"
bash "$SCRIPT_DIR/scripts/setup.sh"

# ---- Step 2: Install model ----
echo -e "${YELLOW}⚙️  Step 2/4: Preparing Ollama & model...${NC}"
bash "$SCRIPT_DIR/scripts/install_model.sh"

# ---- Step 3: Start server ----
echo -e "${YELLOW}⚙️  Step 3/4: Starting FastAPI server...${NC}"
cd "$SCRIPT_DIR"
python3 -m src.server &
SERVER_PID=$!
sleep 3

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start. Check errors above.${NC}"
    exit 1
fi

# ---- Step 4: Tunnel ----
echo -e "${YELLOW}⚙️  Step 4/4: Creating Cloudflare tunnel...${NC}"
bash "$SCRIPT_DIR/scripts/tunnel.sh"

# ---- Keep alive ----
echo ""
echo -e "${GREEN}✅ Gateway is running.${NC}"
echo -e "${GREEN}   Server PID: $SERVER_PID${NC}"
echo ""
wait $SERVER_PID
