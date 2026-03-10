#!/bin/bash
# Launch N autoresearch agents in separate git worktrees.
#
# Usage: ./launch-agents.sh [NUM_AGENTS] [TAG]
# Example: ./launch-agents.sh 2 mar10
#
# This creates worktree directories, shared knowledge files, and prints
# instructions for starting each agent in a separate terminal.

set -e

NUM_AGENTS=${1:-2}
TAG=${2:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}
WORKTREE_DIR="worktrees"
SHARED_DIR="$WORKTREE_DIR/shared"
REPO_ROOT=$(git rev-parse --show-toplevel)

echo "=== Multi-Agent Autoresearch Launcher ==="
echo "Agents:  $NUM_AGENTS"
echo "Tag:     $TAG"
echo "Repo:    $REPO_ROOT"
echo ""

# 1. Create shared knowledge directory
mkdir -p "$SHARED_DIR"

# 2. Initialize shared knowledge files
if [ ! -f "$SHARED_DIR/lessons.md" ]; then
    cat > "$SHARED_DIR/lessons.md" << 'LESSONS_EOF'
# Lessons Learned

Factual experimental findings shared across all agents.
Format: `[AGENT_N] [status] description: val_bpb (delta)`

---

LESSONS_EOF
    echo "Created $SHARED_DIR/lessons.md"
fi

if [ ! -f "$SHARED_DIR/insights.md" ]; then
    cat > "$SHARED_DIR/insights.md" << 'INSIGHTS_EOF'
# Research Insights

Hypotheses, theoretical observations, and promising directions.
Each agent adds insights with their agent number.

---

INSIGHTS_EOF
    echo "Created $SHARED_DIR/insights.md"
fi

# 3. Create agent worktrees
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    AGENT_DIR="$WORKTREE_DIR/agent_$i"
    BRANCH="autoresearch/${TAG}-agent${i}"

    if [ -d "$AGENT_DIR" ]; then
        echo "Agent $i: directory already exists at $AGENT_DIR, skipping."
        continue
    fi

    # Create git worktree on a new branch from current HEAD
    git worktree add "$AGENT_DIR" -b "$BRANCH"

    # Initialize results.tsv with header
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > "$AGENT_DIR/results.tsv"

    # Copy the lock wrapper and program into the worktree
    cp "$REPO_ROOT/train_with_lock.sh" "$AGENT_DIR/"
    chmod +x "$AGENT_DIR/train_with_lock.sh"
    cp "$REPO_ROOT/program.md" "$AGENT_DIR/"

    echo "Agent $i: ready at $AGENT_DIR (branch: $BRANCH)"
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Shared knowledge files:"
echo "  $SHARED_DIR/lessons.md"
echo "  $SHARED_DIR/insights.md"
echo ""

# 4. Spawn agents as background Claude processes
PIDS=()
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    AGENT_DIR="$REPO_ROOT/$WORKTREE_DIR/agent_$i"
    LOG_FILE="$REPO_ROOT/$WORKTREE_DIR/agent_${i}_claude.log"
    PROMPT="You are Sub Agent $i. There are $NUM_AGENTS total agents. Your working directory is $(pwd)/$WORKTREE_DIR/agent_$i/. Read program.md and begin your experiment loop."

    echo "Spawning Agent $i (log: $LOG_FILE)..."
    (cd "$AGENT_DIR" && claude --dangerously-skip-permissions -p "$PROMPT" > "$LOG_FILE" 2>&1) &
    PIDS+=($!)
done

echo ""
echo "=== All $NUM_AGENTS agents launched ==="
echo ""
echo "Agent PIDs: ${PIDS[*]}"
echo ""
echo "Monitor logs:"
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    echo "  tail -f $WORKTREE_DIR/agent_${i}_claude.log"
done
echo ""
echo "Stop all agents:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Clean up worktrees:"
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    echo "  git worktree remove $WORKTREE_DIR/agent_$i"
done
echo "  rm -rf $WORKTREE_DIR"
echo ""

# Wait for all agents (Ctrl+C to stop)
echo "Waiting for agents... (Ctrl+C to stop all)"
trap 'echo ""; echo "Stopping agents..."; kill ${PIDS[*]} 2>/dev/null; wait; echo "Done."; exit 0' INT TERM
wait
