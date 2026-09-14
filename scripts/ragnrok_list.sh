# ragnrok_list.sh — list RAGNAROK-managed files
# Sourced by the `ragnrok` dispatcher (no side effects on source).
#
# Output: a NAME / TYPE / SIZE table over the storage directory, with the
# LOCATION printed once at the bottom. TYPE is derived from the extension
# ("model" for *.gguf, "file" otherwise) — a display label only, NOT a
# classification of the file's role (RAGNAROK does not classify files).

# ── portable file size (GNU stat, then BSD stat) ─────────────────────────────
ragnrok_file_size() {
    local f="$1" sz=""
    if ! sz="$(stat -c '%s' "$f" 2>/dev/null)"; then
        sz="$(stat -f '%z' "$f" 2>/dev/null)" || sz=""
    fi
    if [ -z "$sz" ]; then
        sz="0"
    fi
    printf '%s' "$sz"
}

# ── main ─────────────────────────────────────────────────────────────────────
ragnrok_list_main() {
    local BOLD='\033[1m' DIM='\033[2m' NC='\033[0m'

    local STORAGE
    STORAGE="$(ragnrok_storage_dir)"

    # Header always shown (so the columns are predictable).
    printf '%-32s %-8s %s\n' "NAME" "TYPE" "SIZE"

    # No storage dir yet → empty, not an error.
    if [ ! -d "$STORAGE" ]; then
        echo -e "${DIM}(no files — storage dir does not exist yet)${NC}"
        echo ""
        echo -e "${BOLD}LOCATION:${NC} $STORAGE"
        return 0
    fi

    local count=0 f NAME TYPE BYTES SIZE
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        NAME="$(basename "$f")"
        if [[ "$NAME" == *.gguf ]]; then
            TYPE="model"
        else
            TYPE="file"
        fi
        BYTES="$(ragnrok_file_size "$f")"
        SIZE="$(ragnrok_human_size "$BYTES")"
        printf '%-32s %-8s %s\n' "$NAME" "$TYPE" "$SIZE"
        count=$((count + 1))
    done < <(find "$STORAGE" -maxdepth 1 -type f ! -name '.*' 2>/dev/null | sort)

    if [ "$count" -eq 0 ]; then
        echo -e "${DIM}(empty)${NC}"
    fi

    echo ""
    echo -e "${BOLD}LOCATION:${NC} $STORAGE"
    return 0
}
