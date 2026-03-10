# Multi-Agent Autoresearch Swarm (macOS)

This is a competitive and cooperative multi-agent experiment where multiple LLM subagents conduct independent machine learning research, share insights, and compete to achieve the lowest `val_bpb` on Apple Silicon (MPS).

## System Architecture & Roles

The system consists of one **Master Agent** and **N Subagents** (where N is the total number of subagents specified by the user at launch).

### 1. Master Agent Initialization

If you are the Master Agent, your role is strictly to orchestrate the environment:

1. Receive the target number of subagents (**N**) from the user.
2. Run `./launch-agents.sh N <tag>` to create the worktree structure. This will:
   - Create `worktrees/shared/` with empty `lessons.md` and `insights.md`
   - Create `worktrees/agent_0/` through `worktrees/agent_{N-1}/` as git worktrees on separate branches
   - Initialize `results.tsv` in each agent directory
   - Copy `train_with_lock.sh` into each agent directory
3. Spawn the **N** Subagents. Provide **exactly** this prompt to each subagent: "You are Sub Agent {i}. There are {N} total agents in this swarm. Your working directory is worktrees/agent_{i}/. Read program.md in your directory to understand your rules, constraints, and goals. Begin your experiment loop."

## Experimentation Setup (Subagent)

Before you begin your loop, ensure your workspace is ready:

1. Read the in-scope files in your directory for context (`README.md`, `prepare.py`, `train.py`).
2. Do not modify `prepare.py`. It is read-only and contains the fixed evaluation, data loading, tokenizer, and constants.
3. Verify data exists in `~/.cache/autoresearch/`.
4. Initialize your local `results.tsv` with just the header row (if not already done by launcher).
5. Confirm `../shared/lessons.md` and `../shared/insights.md` exist.

## Rules of Engagement

You run on a shared MPS GPU (Apple Silicon). The training script runs for a fixed time budget of 5 minutes. **Only one agent can train at a time** — use the lock wrapper.

**What you CAN do:**

* Modify your local `train.py` — this is the only code file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**

* Modify `prepare.py`.
* Modify another agent's files.
* Install new packages or add dependencies. Use what's in `pyproject.toml`.
* Modify the evaluation harness (`evaluate_bpb` in `prepare.py`).

**Evaluation:**
The goal is simple: get the lowest `val_bpb`. Since the time budget is fixed (5 minutes), you don't need to worry about training time — it's always 5 minutes. Memory is a soft constraint; some increase is acceptable for meaningful gains, but do not OOM.

**Simplicity criterion:**
All else being equal, simpler is better. A 0.001 val_bpb improvement that adds 20 lines of hacky code is not worth it. A 0.001 val_bpb improvement from deleting code is a huge win.

## MPS GPU Coordination

This Mac has **one shared MPS GPU**. Only one agent can train at a time.

**To run training**, use the lock wrapper instead of calling `uv run train.py` directly:

```bash
./train_with_lock.sh <your_agent_number>
```

This acquires an atomic MPS lock, runs training, and releases the lock automatically (even on crash). If the GPU is busy (another agent is training), the wrapper will wait and retry every 10 seconds.

**While waiting for the GPU**, use your time productively:

1. Read other agents' results: `cat ../agent_*/results.tsv`
2. Read shared knowledge: `cat ../shared/lessons.md`
3. Read shared insights: `cat ../shared/insights.md`
4. Plan your next experiment based on what you've learned
5. Analyze patterns across all agents' results
6. Write hypotheses to `../shared/insights.md`

## Logging & Knowledge Sharing

You are required to maintain three distinct logs to facilitate the swarm's collective intelligence:

### 1. `results.tsv`

Log every experiment here (tab-separated). The headers are:
`commit` \t `val_bpb` \t `memory_gb` \t `status` \t `description`
*(Use 0.000000 for crashes. Status must be `keep`, `discard`, or `crash`)*

### 2. `../shared/lessons.md`

A running log of empirical facts shared across all agents. What worked? What crashed? What blew up memory? Keep this brief and factual so other agents can parse it quickly. **Append** to this file after every experiment. Example:

* *[AGENT_0] [keep] GeLU activation: 1.523 (-0.004 from baseline)*
* *[AGENT_1] [discard] doubled batch size: OOM crash*
* *[AGENT_0] [keep] gradient clipping (1.0): 1.521 (-0.002)*

### 3. `../shared/insights.md`

A higher-level log for your architectural hypotheses. Why do you think a certain mechanism failed? What is the overarching direction you are exploring? Read peers' insights to branch out into uncharted territory instead of overlapping. Format:

```
## [AGENT_N] Hypothesis title
Explanation and reasoning.
Status: untested / promising / dead end
```

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key metric from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## The Experiment Loop

As Agent `i`, you operate autonomously in `worktrees/agent_i/`.

**Initial Step: The Baseline**

* **If you are Agent 0:** Your very first task is to establish the baseline. Do NOT modify `train.py` on your first attempt. Execute the script as-is, evaluate it, and log the initial `val_bpb` in your `results.tsv` (status: `keep`, description: `baseline`) and `../shared/lessons.md`. Once the baseline is logged, enter the loop below.
* **If you are Agent 1 to N-1:** Skip the baseline. You are free to begin radical exploration and modify `train.py` immediately.

**LOOP FOREVER:**

1. **Scout:** Read the `../shared/lessons.md` and `../shared/insights.md`. Incorporate other agents' successes and avoid their failures.
2. **Hypothesize:** Formulate an experimental idea and write your theory in `../shared/insights.md`.
3. **Execute:** Modify `train.py` directly. Git commit your changes.
4. **Run:** Execute `./train_with_lock.sh <your_agent_number>` (this handles MPS lock + redirects output to `run.log`). Do NOT let output flood your context.
5. **Evaluate:** Extract metrics using `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
   * If the run crashed (empty grep), read the stack trace with `tail -n 50 run.log`. Fix minor bugs. If fundamentally broken, discard.
6. **Log:**
   * Record the result in `results.tsv`.
   * Document what happened in `../shared/lessons.md`.
7. **Branch Management:**
   * If `val_bpb` improved (lower), keep the commit and advance.
   * If `val_bpb` is equal or worse, `git reset --hard HEAD~1` to revert to your last good state.

**Timeout**: Each experiment should take ~5 minutes training (+ startup/eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: If it's easy to fix (typo, missing import), fix and re-run. If the idea is fundamentally broken, log "crash" and move on.

**Cross-Agent Intelligence**: Every 3-5 experiments, do a reconnaissance round:

1. Read all agents' results: `for d in ../agent_*/; do echo "=== $d ==="; cat "$d/results.tsv"; done`
2. Read shared lessons and insights
3. Look for ideas that improved val_bpb that you haven't incorporated
4. Avoid techniques that consistently fail across agents

**Exploration Diversity**: To maximize collective progress, diversify your approach. If another agent is exploring architecture changes, focus on optimizer tuning. If another is trying larger models, try smaller models with better hyperparameters. Avoid running the exact same experiment as another agent.

**NEVER STOP:** Do NOT pause to ask the human if you should continue. The human expects you to work indefinitely. If you run out of ideas, read your peers' files, read the code again, combine near-misses, or try radical simplifications. You are an autonomous researcher in a competitive swarm. May the best agent win.
