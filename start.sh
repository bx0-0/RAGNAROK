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
NUM_CTX=16384
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
        --model)
            shift
            MODEL_NAME=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                if [ -z "$MODEL_NAME" ]; then
                    MODEL_NAME="$1"
                else
                    MODEL_NAME="$MODEL_NAME $1"
                fi
                shift
            done
            ;;
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
            echo "  --model <model1> [model2] ...   Model(s) to load"
            echo "                                  Regular: qwen3:8b, llama3.3:70b"
            echo "                                  HuggingFace GGUF: hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
            echo "                                              hf.co/user/repo:Q8_0"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Extract the first model as default
FIRST_MODEL=$(echo "$MODEL_NAME" | awk '{print $1}')

# Convert HF names to display-friendly aliases for banner + logs
# hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q5_K_M → Qwen3.6-35B-A3B-GGUF:UD-Q5_K_M
DISPLAY_MODELS=""
for M in $MODEL_NAME; do
    if [[ "$M" == hf.co/* ]]; then
        SHORT=$(echo "$M" | sed 's|hf\.co/[^/]*/||')
        DISPLAY_MODELS="$DISPLAY_MODELS$SHORT"
    else
        DISPLAY_MODELS="$DISPLAY_MODELS$M"
    fi
    DISPLAY_MODELS="$DISPLAY_MODELS "
done

export MODEL_NAME FIRST_MODEL MAX_CONCURRENT NUM_CTX NUM_PREDICT NUM_BATCH
export FLASH_ATTN NUM_GPU KEEP_ALIVE PORT DEBUG_MODE VERBOSE_LOG
export SERVER_LOG_FILE TUNNEL_LOG_FILE URL_FILE OLLAMA_PULL_LOG REQUEST_LOG_FILE

clear
echo ""

# ─── Rainbow ASCII Banner ───
echo -e "${RED} _  .-')     ('-.                     .-') _    ('-.     _  .-')               .-. .-')   ${NC}"
echo -e "${RED} ( \( -O )   ( OO ).-.                ( OO ) )  ( OO ).-.( \( -O )              \  ( OO )   ${NC}"
echo -e "${GREEN}  ,------.   / . --. /  ,----.    ,--./ ,--,'   / . --. / ,------.  .-'),-----. ,--. ,--.   ${NC}"
echo -e "${GREEN}  |   \`. '  | \-.  \  '  .-./-') |   \ |  |\   | \-.  \  |   \`. '( OO'  .-.  '|  .'   /   ${NC}"
echo -e "${YELLOW}  |  /  | |.-'-'  |  | |  |_( O- )|    \|  | ).-'-'  |  | |  /  | |/   |  | |  ||      /,  ${NC}"
echo -e "${YELLOW}  |  |_.' | \| |_.'  | |  | .--, \|  .     |/  \| |_.'  | |  |_.' |\_) |  |\|  ||     ' _) ${NC}"
echo -e "${BLUE}  |  .  '.'  |  .-.  |(|  | '. (_/|  |\    |    |  .-.  | |  .  '.'  \ |  | |  ||  .   \   ${NC}"
echo -e "${BLUE}  |  |\  \   |  | |  | |  '--'  | |  | \   |    |  | |  | |  |\  \    \`'  '-'  '|  |\   \  ${NC}"
echo -e "${MAGENTA}  \`--' '--'  \`--' \`--'  \`------'  \`--'  \`--'    \`--' \`--' \`--' '--'     \`-----' \`--' '--' ${NC}"
echo ""
echo -e "  ${CYAN}${BOLD}GPU Model Gateway${NC}"
echo -e "  ${DIM}Run Ollama on Kaggle & Colab GPUs with a public OpenAI-compatible API${NC}"
echo ""
echo -e "  ${DIM}Models: ${GREEN}${DISPLAY_MODELS}${DIM}    |    Port: ${YELLOW}${PORT}${NC}"
echo -e "  ${DIM}Context: ${GREEN}${NUM_CTX}${DIM}    |    GPU: ${YELLOW}${NUM_GPU}${DIM}    |    Flash: ${YELLOW}${FLASH_ATTN}${NC}"
echo ""

# ─── Step 1 ───
echo -e "${BOLD}${WHITE}[1/4]${NC} ${DIM}Installing dependencies...${NC}"
bash "$SCRIPT_DIR/scripts/setup.sh"

# ─── Step 2 ───
echo -e "${BOLD}${WHITE}[2/4]${NC} ${DIM}Preparing Ollama & model(s)...${NC}"
bash "$SCRIPT_DIR/scripts/install_model.sh"

# ─── Step 3 ───
echo -e "${BOLD}${WHITE}[3/4]${NC} ${DIM}Starting FastAPI server...${NC}"
cd "$SCRIPT_DIR"
pkill -f "src.server" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

python3 -m src > "${SERVER_LOG_FILE}" 2>&1 &
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
    tail -30 "${SERVER_LOG_FILE}" 2>/dev/null
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo -e " ${GREEN}✅${NC}"

# ─── Step 4 ───
echo -e "${BOLD}${WHITE}[4/4]${NC} ${DIM}Creating Cloudflare tunnel...${NC}"
bash "$SCRIPT_DIR/scripts/tunnel_orchestrate.sh"
