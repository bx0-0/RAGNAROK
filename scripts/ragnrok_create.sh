# ragnrok_create.sh — create an Ollama model from a local GGUF
# Sourced by the `ragnrok` dispatcher (no side effects on source).
#
# Two modes:
#
#   normal : ragnrok create model.gguf [--parser NAME] [--render NAME] [--name NAME]
#   mtp    : ragnrok create --mtp model.gguf [assoc.gguf ...] [--parser NAME] [--render NAME] [--name NAME]
#
# Semantics (per RAGNAROK design):
#   * The first positional file is ALWAYS the main model.
#   * In --mtp mode, every subsequent positional is a *generic* associated
#     file (one or many). RAGNAROK does NOT classify them (vision / audio /
#     MTP / adapter) and does NOT inspect GGUF metadata — they are passed
#     through unchanged.
#   * --parser NAME / --render NAME each take an explicit value (a renderer/parser name).
#   * Positional collection stops at the next `--...` option.
#
# Reuse (NOT duplicated in Bash):
#   * Modelfile -> `ollama create` goes through the EXISTING
#     src.model_manager.ModelManager.create_from_modelfile() (the proven path
#     that accepts the custom RENDERER / PARSER directives).
#
# Modelfile produced:
#   FROM <main-model-path>
#   RENDERER <value>    (only if --render VALUE)
#   PARSER <value>      (only if --parser VALUE)
#   FROM <assoc-1-path> (one per associated file, in order — generic pass-through)
#
# Design note (documented assumption): the only Modelfile directive in this
# codebase that references a weight file is `FROM` (see repair_manager /
# model_manager and the repair tests). Associated files are therefore appended
# as additional `FROM` lines so the existing `ollama create` path sees them.
# Behavior depends on the RAGNAROK Ollama build accepting that form.
#
# Design note (value flags): --parser / --render each take an explicit value
# (a renderer/parser name), matching the repair API (src/models/repair.py).
# Values are validated as simple identifiers (alnum, dot, dash, underscore).
#

# ── derive an Ollama model name from a file path ─────────────────────────────
# basename, strip a trailing .gguf (case-insensitive), lowercase.
ragnrok_create_model_name() {
    local base
    base="$(basename "$1")"
    base="${base%.gguf}"
    base="${base%.GGUF}"
    base="${base%.Gguf}"
    base="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
    printf '%s\n' "$base"
}

# ── validate a --parser / --render value (simple identifier) ────────────────
# Mirrors src/models/repair.py:_validate_token. Returns 0 if OK, 1 otherwise.
ragnrok_create_check_value() {
    local v="$1"
    if [[ ! "$v" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        echo -e "${RED}Error: invalid value for $OPT: $v${NC}" >&2
        echo -e "${DIM}  expected a simple identifier (alnum, dot, dash, underscore)${NC}" >&2
        return 1
    fi
    return 0
}

# ── main ─────────────────────────────────────────────────────────────────────
ragnrok_create_main() {
    local RED='\033[0;31m' GREEN='\033[0;32m'
    local DIM='\033[2m' BOLD='\033[1m' NC='\033[0m'

    local MTP=0 PARSER="" RENDER="" NAME=""
    local FILES=()

    # ── parse args (positionals stop at next `--...` option) ─────────────
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mtp)      MTP=1; shift ;;
            --parser)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}Error: --parser requires a value.${NC}" >&2
                    return 1
                fi
                if [[ "$2" == --* ]]; then
                    echo -e "${RED}Error: --parser requires a value (got '$2').${NC}" >&2
                    return 1
                fi
                PARSER="$2"; shift 2
                ;;
            --render)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}Error: --render requires a value.${NC}" >&2
                    return 1
                fi
                if [[ "$2" == --* ]]; then
                    echo -e "${RED}Error: --render requires a value (got '$2').${NC}" >&2
                    return 1
                fi
                RENDER="$2"; shift 2
                ;;
            --name)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}Error: --name requires a value.${NC}" >&2
                    return 1
                fi
                NAME="$2"; shift 2 ;;
            --help|-h)
                echo "Usage:"
                echo "  ragnrok create model.gguf [--parser NAME] [--render NAME] [--name NAME]"
                echo "  ragnrok create --mtp model.gguf [assoc.gguf ...] [--parser NAME] [--render NAME] [--name NAME]"
                echo ""
                echo "  First file = main model. Extra files (--mtp) = associated files."
                echo "  --parser NAME / --render NAME set the renderer/parser (value required)."
                echo "  --name sets the Ollama model name."
                exit 0
                ;;
            -*)
                echo -e "${RED}Error: unknown option: $1${NC}" >&2
                return 1
                ;;
            *)
                FILES+=("$1"); shift ;;
        esac
    done

    # ── validate --parser / --render values ──────────────────────────────
    if [ -n "$PARSER" ]; then
        OPT=--parser ragnrok_create_check_value "$PARSER" || return 1
    fi
    if [ -n "$RENDER" ]; then
        OPT=--render ragnrok_create_check_value "$RENDER" || return 1
    fi

    # ── validation: at least one file (the main model) ────────────────────
    if [ ${#FILES[@]} -eq 0 ]; then
        echo -e "${RED}Error: no model file provided.${NC}" >&2
        echo "  Usage: ragnrok create model.gguf [--parser NAME] [--render NAME]" >&2
        echo "         ragnrok create --mtp model.gguf [assoc.gguf ...] [--parser NAME] [--render NAME]" >&2
        return 1
    fi

    # ── resolve every file to a real path (storage dir first, then CWD) ──
    local RESOLVED=() f rp
    for f in "${FILES[@]}"; do
        if rp="$(ragnrok_resolve_file "$f")"; then
            RESOLVED+=("$rp")
        else
            local S
            S="$(ragnrok_storage_dir)"
            echo -e "${RED}Error: file not found: $f${NC}" >&2
            echo -e "${DIM}  looked in: $S  and  $PWD${NC}" >&2
            return 1
        fi
    done

    local MAIN="${RESOLVED[0]}"
    local MODEL_NAME
    if [ -n "$NAME" ]; then
        MODEL_NAME="$NAME"
    else
        MODEL_NAME="$(ragnrok_create_model_name "$MAIN")"
    fi

    # ── build the Modelfile ────────────────────────────────────────────────
    local MF="FROM ${MAIN}"
    [ -n "$RENDER" ] && MF+=$'\n'"RENDERER ${RENDER}"
    [ -n "$PARSER" ] && MF+=$'\n'"PARSER ${PARSER}"
    local i
    for (( i=1; i<${#RESOLVED[@]}; i++ )); do
        MF+=$'\n'"FROM ${RESOLVED[$i]}"
    done

    # ── echo a concise plan ────────────────────────────────────────────────
    echo -e "${BOLD}Ollama model:${NC} ${MODEL_NAME}"
    echo -e "${BOLD}Modelfile:${NC}"
    local line
    while IFS= read -r line; do
        echo -e "  ${DIM}${line}${NC}"
    done < <(printf '%s\n' "$MF")
    echo ""

    # ── invoke the EXISTING model-generation mechanism (Python) ──────────
    # Reuse src.model_manager.ModelManager.create_from_modelfile (proven
    # `ollama create -f` path). NOT re-implemented in Bash.
    # The dispatcher exports RAGNROK_REPO_ROOT (repo root). Required so the
    # existing src.* modules are importable.
    if [ -z "${RAGNROK_REPO_ROOT:-}" ]; then
        echo -e "${RED}Error: RAGNROK_REPO_ROOT is not set (use the 'ragnrok' dispatcher).${NC}" >&2
        return 1
    fi
    local PYTHONPATH="${PYTHONPATH:-}"
    PYTHONPATH="${RAGNROK_REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

    local rc=0
    PYTHONPATH="$PYTHONPATH" python3 -c '
import sys
try:
    import asyncio
    import ollama
    from src.model_manager import ModelManager, CreateError
except Exception as e:
    print(f"Error: could not load ragnrok model management: {e}", file=sys.stderr)
    sys.exit(3)

name, modelfile = sys.argv[1], sys.argv[2]
async def main():
    mm = ModelManager(ollama.AsyncClient())
    await mm.create_from_modelfile(name, modelfile)

try:
    asyncio.run(main())
except CreateError as e:
    detail = (e.stderr or e.stdout or "").strip()
    print(f"Error: {e}", file=sys.stderr)
    if detail:
        print(detail, file=sys.stderr)
    sys.exit(1)
' "$MODEL_NAME" "$MF" || rc=$?

    if [ "$rc" -ne 0 ]; then
        echo -e "${RED}Model creation failed (exit ${rc}).${NC}" >&2
        return "$rc"
    fi

    echo -e "${GREEN}Created Ollama model: ${MODEL_NAME}${NC}"
    return 0
}
