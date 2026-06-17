#!/bin/bash
#
# Install / pull the selected Ollama model(s)
#

set -e

# MODEL_NAME can be space-separated list
MODELS="$MODEL_NAME"

echo "  ├─ Stopping any existing Ollama..."
if pgrep ollama >/dev/null 2>&1; then
    pkill ollama
    sleep 2
fi

echo "  ├─ Starting Ollama serve..."
if ! pgrep ollama >/dev/null 2>&1; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
else
    echo "  │  ℹ️  Ollama already running"
fi

# Pull each model in the list
for MODEL in $MODELS; do
    echo ""
    echo "  ├─ Pulling model: $MODEL"
    # Check if model exists — works for both regular and HF names (hf.co/...)
    MODEL_EXISTS=$(ollama list 2>/dev/null | awk '{print $1}' | grep -Fx "$MODEL" || true)
    if [ -n "$MODEL_EXISTS" ]; then
        echo "  │  ℹ️  Model already cached"
    else
        ollama pull "$MODEL" > /tmp/ollama-pull.log 2>&1 &
        PULL_PID=$!
        printf "  │  Downloading "
        while kill -0 $PULL_PID 2>/dev/null; do
            printf "."
            sleep 3
        done
        wait $PULL_PID
        echo ""
        echo "  │  ✅ Model downloaded"
    fi
done

echo ""
echo "  └─ Done."
