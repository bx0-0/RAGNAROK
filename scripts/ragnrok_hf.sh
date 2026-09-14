# ragnrok_hf.sh — Hugging Face downloads into RAGNAROK managed storage
# Sourced by the `ragnrok` dispatcher (no side effects on source).
#
# Usage:
#   ragnrok hf --download REPO [FILE ...] [--max-workers N]
#
# A faithful translation of the proven `hf download` CLI form:
#   hf download <REPO> [FILE ...] --local-dir <dir> [--max-workers N]
#
# REPO   = "owner/repo-id" (or a full huggingface.co URL; the host is stripped)
# FILE   = one or more repo-relative filenames. Typical MTP case: the --mtp
#          model plus its mmproj projector, both in the same repo. If no FILE
#          is given, the whole repo is fetched.
#
# Downloads land directly in the RAGNAROK managed storage directory via
# `hf download --local-dir <storage>` (no copy across filesystems).
#

# ── reduce a repo ref (path or URL) to "owner/repo-id" ───────────────────────
ragnrok_hf_repo_id() {
    local r="$1"
    r="${r#https://huggingface.co/}"
    r="${r#http://huggingface.co/}"
    local first="${r%%/*}"
    local second
    second="$(printf '%s\n' "$r" | cut -d/ -f2)"
    if [ -n "$second" ]; then
        printf '%s/%s\n' "$first" "$second"
    else
        printf '%s\n' "$first"
    fi
}

# ── reduce a file ref (path or URL) to its repo-relative name ───────────────
ragnrok_hf_file_name() {
    printf '%s\n' "${1##*/}"
}

# ── main ─────────────────────────────────────────────────────────────────────
ragnrok_hf_main() {
    local RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
    local CYAN='\033[0;36m' DIM='\033[2m' BOLD='\033[1m' NC='\033[0m'

    local DOWNLOAD=0 REPO="" MAXWORKERS=""
    local FILES=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --download)
                DOWNLOAD=1
                shift
                if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                    REPO="$1"; shift
                fi
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    FILES+=("$1"); shift
                done
                ;;
            --max-workers)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}Error: --max-workers requires a number.${NC}" >&2
                    return 1
                fi
                MAXWORKERS="$2"; shift 2
                ;;
            --help|-h)
                echo "Usage: ragnrok hf --download REPO [FILE ...] [--max-workers N]"
                echo ""
                echo "  Download one or more files from a Hugging Face repo into"
                echo "  RAGNAROK managed storage (via the 'hf download' CLI)."
                echo ""
                echo "  REPO   owner/repo-id  (or a full huggingface.co URL)"
                echo "  FILE   repo-relative filename(s); omit to fetch the whole repo"
                exit 0
                ;;
            -*)
                echo -e "${RED}Error: unknown option: $1${NC}" >&2
                echo "  Usage: ragnrok hf --download REPO [FILE ...] [--max-workers N]" >&2
                return 1
                ;;
            *)
                if [ "$DOWNLOAD" -eq 0 ] && [ -z "$REPO" ]; then
                    REPO="$1"; shift
                else
                    FILES+=("$1"); shift
                fi
                ;;
        esac
    done

    if [ "$DOWNLOAD" -ne 1 ]; then
        echo -e "${RED}Error: ragnrok hf requires --download${NC}" >&2
        echo "  Usage: ragnrok hf --download REPO [FILE ...] [--max-workers N]" >&2
        return 1
    fi

    if [ -z "$REPO" ]; then
        echo -e "${RED}Error: no repository was provided.${NC}" >&2
        echo "  Usage: ragnrok hf --download REPO [FILE ...] [--max-workers N]" >&2
        return 1
    fi

    if ! command -v hf >/dev/null 2>&1; then
        echo -e "${RED}Error: 'hf' command not found.${NC}" >&2
        echo "  Install it with:  pip install -U \"huggingface_hub[cli]\"" >&2
        return 1
    fi

    local REPOID
    REPOID="$(ragnrok_hf_repo_id "$REPO")"

    local STORAGE
    STORAGE="$(ragnrok_ensure_storage)" || return 1
    echo -e "${BOLD}Storage:${NC} ${DIM}$STORAGE${NC}"
    echo ""

    local NORM_FILES=() f
    for f in "${FILES[@]}"; do
        NORM_FILES+=("$(ragnrok_hf_file_name "$f")")
    done

    local ARGS=(download "$REPOID" --local-dir "$STORAGE")
    [ ${#NORM_FILES[@]} -gt 0 ] && ARGS+=("${NORM_FILES[@]}")
    [ -n "$MAXWORKERS" ] && ARGS+=(--max-workers "$MAXWORKERS")

    echo -e "${BOLD}${CYAN}↓${NC} ${DIM}hf ${ARGS[*]}${NC}"
    local rc=0
    hf "${ARGS[@]}" || rc=$?
    echo ""

    if [ "$rc" -ne 0 ]; then
        echo -e "${YELLOW}hf download failed (exit ${rc}) — file may be missing or network error.${NC}" >&2
        return "$rc"
    fi

    echo -e "${GREEN}Download complete.${NC} Storage: ${DIM}$STORAGE${NC}"
    return 0
}
