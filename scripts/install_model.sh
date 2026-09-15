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

# Wait until the server actually accepts commands before trusting
# `ollama list` - a freshly (re)started server can take a few seconds, and
# an empty list here would make us "pull" models that are already in the
# local store (e.g. ones created by ragnrok create).
READY=0
for i in $(seq 1 "${OLLAMA_READY_WAIT:-30}"); do
    if ollama list >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" -ne 1 ]; then
    echo "  Ollama server did not become ready (waited ${OLLAMA_READY_WAIT:-30}s)"
    exit 1
fi

# Pull each model in the list
for MODEL in $MODELS; do
    echo ""
    echo "  ├─ Pulling model: $MODEL"
    # Check if model exists - works for both regular and HF names (hf.co/...).
    # Locally created models are listed as "name:latest", so strip a trailing
    # ":latest" (HF refs such as hf.co/owner/repo:main are untouched) so they
    # are recognised as already present instead of being pulled again.
    MODEL_EXISTS=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' | sed 's/:latest$//' | grep -Fx "$MODEL" || true)
    if [ -n "$MODEL_EXISTS" ]; then
        echo "  │  ℹ️  Model already cached"
    else
        ollama pull "$MODEL" > "${OLLAMA_PULL_LOG}" 2>&1 &
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

    # Create a short alias for Hugging Face models so clients use clean names
    if [[ "$MODEL" == hf.co/* ]]; then
        SHORT_NAME=$(echo "$MODEL" | sed 's|hf\.co/[^/]*/||')
        ALIAS_EXISTS=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' | sed 's/:latest$//' | grep -Fx "$SHORT_NAME" || true)
        if [ -z "$ALIAS_EXISTS" ]; then
            echo "  ├─ Creating alias: $SHORT_NAME"
            if ollama cp "$MODEL" "$SHORT_NAME" > /dev/null 2>&1; then
                echo "  │  ✅ Alias created (shares data, no extra disk space)"
            else
                echo "  │  ⚠️  Alias creation failed — will use full name"
            fi
        else
            echo "  │  ℹ️  Alias already exists: $SHORT_NAME"
        fi
    fi
done

echo ""
echo "  └─ Done."
