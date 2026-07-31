#!/bin/bash
#
# Install TTS engine models (OmniVoice + Inflect)
# Called from start.sh — errors are non-fatal so the server still starts
#
# Inflect: install separately on Kaggle:
#   pip install --upgrade huggingface_hub
#   hf download owensong/Inflect-Micro-v2 --local-dir Inflect-Micro-v2
#   cd Inflect-Micro-v2 \u0026\u0026 pip install -r requirements.txt

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${WHITE}[2.5/4]${NC} ${DIM}Preparing TTS models...${NC}"

if [ "$TTS_ENABLED" != "true" ] \u0026\u0026 [ "$TTS_ENABLED" != "True" ]; then
    echo "  └─ Skipped (TTS disabled)"
    exit 0
fi

# --- OmniVoice model download ---
echo ""
echo "  ├─ Downloading OmniVoice model ..."

omni_ok=1
python3 << 'PYEOF'
import os, sys
try:
    from huggingface_hub import snapshot_download
    cache = snapshot_download(
        repo_id="k2-fsa/OmniVoice",
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
    )
    print(f"  │  ✅ OmniVoice model cached")
except Exception as e:
    print(f"  │  ⚠️  OmniVoice download failed: {e}")
    sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
    omni_ok=0
fi

if [ "$omni_ok" -eq 1 ]; then
    echo "  │  ✅ OmniVoice ready"
else
    echo -e "  │  ${YELLOW}ℹ️  OmniVoice model download failed — TTS endpoint will return an error until models are installed${NC}"
fi

# --- Inflect model download ---
echo ""
variant="${TTS_INFLECT_VARIANT:-nano}"
repo="owensong/Inflect-Nano-v2"
if [ "$variant" = "micro" ]; then
    repo="owensong/Inflect-Micro-v2"
fi

echo "  ├─ Downloading Inflect ${variant} model ..."

inflect_ok=1
python3 << PYEOF
import os, sys
try:
    from huggingface_hub import snapshot_download
    cache = snapshot_download(
        repo_id="$repo",
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
    )
    print(f"  │  ✅ Inflect ${variant} model cached")
except Exception as e:
    print(f"  │  ⚠️  Inflect download failed: {e}")
    sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
    inflect_ok=0
fi

if [ "$inflect_ok" -eq 1 ]; then
    echo "  │  ✅ Inflect ready"
else
    echo -e "  │  ${YELLOW}ℹ️  Inflect model download failed — TTS endpoint will return an error until models are installed${NC}"
fi

echo ""
echo "  └─ Done."
