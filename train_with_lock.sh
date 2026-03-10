#!/bin/bash
# MPS GPU lock wrapper for multi-agent autoresearch on macOS.
# Only one agent can train at a time on Apple Silicon's shared MPS GPU.
#
# Usage: ./train_with_lock.sh <AGENT_ID>
# Example: ./train_with_lock.sh 0

set -e

AGENT_ID=${1:?"Usage: ./train_with_lock.sh <AGENT_ID>"}
LOCK_DIR="../.mps_lock.d"
HOLDER_FILE="$LOCK_DIR/holder"
MAX_WAIT=1800       # 30 minutes max wait
POLL_INTERVAL=10    # check every 10 seconds
STALE_THRESHOLD=600 # force-release locks older than 10 minutes

cleanup() {
    rm -rf "$LOCK_DIR" 2>/dev/null
    echo "[agent_$AGENT_ID] MPS lock released."
}

# --- Wait for lock ---
elapsed=0
while true; do
    # Try atomic lock acquire (mkdir is atomic on POSIX)
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        break
    fi

    # Lock exists — check if stale
    if [ -f "$HOLDER_FILE" ]; then
        lock_time=$(awk '{print $2}' "$HOLDER_FILE" 2>/dev/null || echo 0)
        now=$(date +%s)
        age=$((now - lock_time))
        holder=$(awk '{print $1}' "$HOLDER_FILE" 2>/dev/null || echo "unknown")

        if [ "$age" -gt "$STALE_THRESHOLD" ]; then
            echo "[agent_$AGENT_ID] Stale lock detected (held by $holder for ${age}s), forcing release."
            rm -rf "$LOCK_DIR"
            continue
        fi

        if [ $((elapsed % 60)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
            echo "[agent_$AGENT_ID] Waiting for MPS GPU (held by $holder for ${age}s)..."
        fi
    fi

    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "[agent_$AGENT_ID] TIMEOUT: waited ${MAX_WAIT}s for MPS lock."
        exit 1
    fi

    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

# --- Lock acquired ---
echo "agent_$AGENT_ID $(date +%s)" > "$HOLDER_FILE"
trap cleanup EXIT
echo "[agent_$AGENT_ID] MPS lock acquired, starting training..."

# --- Run training ---
uv run train.py > run.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[agent_$AGENT_ID] Training exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
