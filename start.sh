#!/bin/bash
#
# RAGNAROK — Main Launcher
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
WHITE='\033[1;37m'
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
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

export MODEL_NAME MAX_CONCURRENT NUM_CTX NUM_PREDICT NUM_BATCH
export FLASH_ATTN NUM_GPU KEEP_ALIVE PORT DEBUG_MODE VERBOSE_LOG

clear
echo ""

# ─── Rainbow ASCII Banner ───
# Using \x60 for backticks to avoid bash command-substitution parsing errors
echo -e "${RED} _  .-')     ('-.                     .-') _    ('-.     _  .-')               .-. .-')   ${NC}"
echo -e "${RED} ( \( -O )   ( OO ).-.                ( OO ) )  ( OO ).-.( \( -O )              \  ( OO )   ${NC}"
echo -e "${GREEN}  ,------.   / . --. /  ,----.    ,--./ ,--,'   / . --. / ,------.  .-'),-----. ,--. ,--.   ${NC}"
echo -e "${GREEN}  |   /\x60. '  | \-.  \  '  .-./-') |   \ |  |\   | \-.  \  |   /\x60. '( OO'  .-.  '|  .'   /   ${NC}"
echo -e "${YELLOW}  |  /  | |.-'-'  |  | |  |_( O- )|    \|  | ).-'-'  |  | |  /  | |/   |  | |  ||      /,  ${NC}"
echo -e "${YELLOW}  |  |_.' | \| |_.'  | |  | .--, \|  .     |/  \| |_.'  | |  |_.' |\_) |  |\|  ||     ' _) ${NC}"
echo -e "${BLUE}  |  .  '.'  |  .-.  |(|  | '. (_/|  |\    |    |  .-.  | |  .  '.'  \ |  | |  ||  .   \   ${NC}"
echo -e "${BLUE}  |  |\  \   |  | |  | |  '--'  | |  | \   |    |  | |  | |  |\  \    \x60'  '-'  '|  |\   \  ${NC}"
echo -e "${MAGENTA}  \x60--' '--'  \x60--' \x60--'  \x60------'  \x60--'  \x60--'    \x60--' \x60--' \x60--' '--'     \x60-----' \x60--' '--' ${NC}"
echo ""
echo -e "  ${DIM}Model: ${GREEN}${MODEL_NAME}${DIM}    |    Port: ${YELLOW}${PORT}${NC}"
echo -e "  ${DIM}Context: ${GREEN}${NUM_CTX}${DIM}    |    GPU: ${YELLOW}${NUM_GPU}${DIM}    |    Threads: ${GREEN}${MAX_CONCURRENT}${DIM}    |    Flash: ${YELLOW}${FLASH_ATTN}${NC}"
echo ""

# ─── Step 1 ───
echo -e "${BOLD}${WHITE}[1/4]${NC} ${DIM}Installing dependencies...${NC}"
bash "$SCRIPT_DIR/scripts/setup.sh"

# ─── Step 2 ───
echo -e "${BOLD}${WHITE}[2/4]${NC} ${DIM}Preparing Ollama & model...${NC}"
bash "$SCRIPT_DIR/scripts/install_model.sh"

# ─── Step 3 ───
echo -e "${BOLD}${WHITE}[3/4]${NC} ${DIM}Starting FastAPI server...${NC}"
cd "$SCRIPT_DIR"
pkill -f "src.server" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

python3 -m src.server > /tmp/gateway-server.log 2>&1 &
SERVER_PID=$!
echo -e "  ${DIM}PID: ${YELLOW}${SERVER_PID}${NC}"

if [ "$VERBOSE_LOG" = "True" ] || [ "$VERBOSE_LOG" = "true" ]; then
    echo -e "  ${DIM}[verbose-log enabled — request log appears after tunnel setup]${NC}"
fi

echo -ne "  ${DIM}Waiting for server${NC}"
READY=0
for i in $(seq 1 30); do
    sleep 2
    printf "${CYAN}.${NC}"
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
        READY=1
        break
    fi
done
if [ "$READY" -ne 1 ]; then
    echo ""
    echo -e "  ${RED}❌ Server failed to start${NC}"
    echo -e "  ${YELLOW}Last 30 lines of server log:${NC}"
    tail -30 /tmp/gateway-server.log 2>/dev/null
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo -e " ${GREEN}✅${NC}"

# ─── Step 4 ───
echo -e "${BOLD}${WHITE}[4/4]${NC} ${DIM}Creating Cloudflare tunnel...${NC}"
bash "$SCRIPT_DIR/scripts/tunnel.sh"
