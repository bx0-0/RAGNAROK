#!/bin/bash
#
# Install TTS engine models (OmniVoice + Inflect)
# Called from start.sh — errors are non-fatal so the server still starts
#

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${WHITE}[2.5/4]${NC} ${DIM}Preparing TTS models...${NC}"

if [ "$TTS_ENABLED" != "true" ] && [ "$TTS_ENABLED" != "True" ]; then
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

# --- Inflect model download & install ---
echo ""
variant="${TTS_INFLECT_VARIANT:-nano}"
repo="owensong/Inflect-Nano-v2"
if [ "$variant" = "micro" ]; then
    repo="owensong/Inflect-Micro-v2"
fi

echo "  ├─ Installing Inflect ${variant} ..."

inflect_ok=1
INSTALL_DIR="/kaggle/working/Inflect-${variant^}"
python3 << PYEOF
import os, sys, subprocess
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    install_dir = "$INSTALL_DIR"
    snapshot_download(repo_id="$repo", local_dir=install_dir)
    print(f"  │  ✅ Inflect ${variant} downloaded to {install_dir}")
except Exception as e:
    print(f"  │  ⚠️  Inflect download failed: {e}")
    sys.exit(1)
PYEOF
if [ $? -eq 0 ]; then
    echo "  │  Installing Inflect dependencies ..."
    cd "$INSTALL_DIR" && pip install -q -r requirements.txt 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  │  ✅ Inflect ${variant} installed"
        # Write model path so the engine can find it
        echo "$INSTALL_DIR" > /tmp/inflect_model_dir.txt
    else
        echo -e "  │  ${YELLOW}⚠️  pip install failed — Inflect may not work${NC}"
        inflect_ok=0
    fi
else
    inflect_ok=0
    echo -e "  │  ${YELLOW}⚠️  Inflect download failed — TTS endpoint will return an error${NC}"
fi

echo ""
echo "  └─ Done."