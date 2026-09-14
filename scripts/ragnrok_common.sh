# ragnrok_common.sh — shared storage-path helpers
# Sourced by ragnrok_hf.sh, ragnrok_list.sh, ragnrok_create.sh
#
# Determines the RAGNAROK managed storage directory based on environment:
#   Kaggle  → /kaggle/working/.ragnrok/   (persistent workspace, ~100GB)
#   Colab   → /tmp/.ragnrok/              (large temp area on Colab)
#   Local   → $HOME/.ragnrok/             (standard user data dir)
#
# Rationale:
#   install_tts.sh already uses /kaggle/working/ for Kaggle model data.
#   On Colab, /tmp is the large scratch area (home is small).
#   On local Linux, $HOME is the natural user-data location.
#

# ── resolve storage root ─────────────────────────────────────────────────────
ragnrok_storage_dir() {
    # Allow explicit override via env var (useful for testing)
    if [ -n "${RAGNROK_STORAGE_DIR:-}" ]; then
        printf '%s\n' "$RAGNROK_STORAGE_DIR"
        return
    fi

    # Kaggle: /kaggle/working is the persistent workspace
    if [ -d "/kaggle/working" ]; then
        printf '%s\n' "/kaggle/working/.ragnrok"
        return
    fi

    # Colab: /tmp is the large area (home is small)
    if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d "/opt/conda" ]; then
        printf '%s\n' "/tmp/.ragnrok"
        return
    fi

    # Local / fallback
    printf '%s\n' "${HOME:-/tmp}/.ragnrok"
}

# ── ensure storage dir exists, print path ───────────────────────────────────
ragnrok_ensure_storage() {
    local dir
    dir="$(ragnrok_storage_dir)"
    if ! mkdir -p "$dir" 2>/dev/null; then
        echo "  ERROR: cannot create storage dir: $dir" >&2
        return 1
    fi
    printf '%s\n' "$dir"
}

# ── human-readable size ─────────────────────────────────────────────────────
ragnrok_human_size() {
    local bytes="$1"
    if [ "$bytes" -ge 1073741824 ] 2>/dev/null; then
        awk "BEGIN {printf \"%.1f GB\", $bytes/1073741824}"
    elif [ "$bytes" -ge 1048576 ] 2>/dev/null; then
        awk "BEGIN {printf \"%.1f MB\", $bytes/1048576}"
    elif [ "$bytes" -ge 1024 ] 2>/dev/null; then
        awk "BEGIN {printf \"%.1f KB\", $bytes/1024}"
    else
        printf '%s B' "$bytes"
    fi
}

# ── resolve a file reference to a full path ─────────────────────────────────
# Accepts: absolute path, relative path, or bare filename.
# Bare names are resolved against the storage dir first, then CWD.
# Prints the resolved path on success; returns 1 on failure.
ragnrok_resolve_file() {
    local ref="$1"
    local dir

    if [[ "$ref" == /* ]]; then
        # absolute
        if [ -f "$ref" ]; then printf '%s\n' "$ref"; else return 1; fi
    elif [[ "$ref" == *"/"* ]]; then
        # relative path
        if [ -f "$ref" ]; then printf '%s\n' "$ref"; else return 1; fi
    else
        # bare name — storage dir first, then CWD
        dir="$(ragnrok_storage_dir)"
        if [ -f "$dir/$ref" ]; then
            printf '%s\n' "$dir/$ref"
        elif [ -f "./$ref" ]; then
            printf '%s\n' "./$ref"
        else
            return 1
        fi
    fi
}
