#!/bin/bash
#
# Extract public Cloudflare URL from tunnel logs
# Usage: source tunnel_url_extract.sh   (functions only, not standalone)
#

extract_tunnel_url() {
    local logfile="${1:-$TUNNEL_LOG_FILE}"
    grep -oP 'https://[a-zA-Z0-9_\-]+\.trycloudflare\.com' "$logfile" 2>/dev/null | tail -1
}

write_url_file() {
    local url="$1"
    echo "$url" > "$URL_FILE"
}
