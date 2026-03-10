# I Let an AI Agent Run 100 Experiments on My MacBook Overnight — Here's How You Can Too

**TL;DR:** Andrej Karpathy's `autoresearch` lets an AI agent autonomously optimize language model training while you sleep. The macOS fork brings this to Apple Silicon Macs. Here's how the architecture works and how to set it up.

---

In March 2026, Andrej Karpathy released something that quietly shifted how we think about ML research. Not a new model. Not a benchmark. A **self-running research lab** — one that fits on a single GPU and runs while you sleep.

The project is called **autoresearch**: give an AI agent a small but real LLM training setup, let it modify the code, train for 5 minutes, check if the result improved, keep or discard, and repeat. You wake up to a log of experiments and a better model.

The catch? The original repo required an NVIDIA GPU. If you're on a Mac — like many of us — you were out of luck.

Until **autoresearch-macos**.

---

## What Is Autoresearch?

At its core, autoresearch is an **autonomous experimentation loop**. You point an AI coding agent (Claude Code, Cursor, Codex, etc.) at the repo, and it:

1. Reads the instructions in `program.md`
2. Modifies `train.py` — changing architecture, hyperparameters, optimizer settings, batch size, anything
3. Runs a training experiment (fixed 5-minute budget)
4. Evaluates using `val_bpb` (validation bits per byte) — a vocab-size-independent metric
5. If the result improved → commits the change and moves on
6. If not → discards and tries something different
7. **Repeats indefinitely**

At ~12 experiments per hour, you get roughly **100 experiments overnight**. No human in the loop.

As Karpathy wrote:

> *"Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies... This repo is the story of how it all began."*

---

## The Architecture: Elegant Simplicity

The genius of autoresearch is its **deliberate constraints**. Only three files matter:

### `prepare.py` — The Fixed Foundation (Read-Only)

This file is never modified. It handles:

- **Data download**: Fetches TinyStories-style data shards from HuggingFace (~10 training shards + 1 pinned validation shard)
- **Tokenizer training**: Builds a BPE tokenizer with 8,192 tokens using rustbpe
- **Evaluation utilities**: The validation metric (BPB) lives here, ensuring fair comparisons across experiments
- **Constants**: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300s`, `EVAL_TOKENS=40×524K`

By keeping evaluation fixed, every experiment is measured on the same yardstick.

### `train.py` — The Playground (Agent Edits This)

This is where all the action happens. It contains the **entire training pipeline** in a single file:

**The Model (GPT Architecture):**
- Transformer with configurable depth and width
- Rotary Position Embeddings (RoPE)
- Group Query Attention (GQA) support
- Value Residual connections (ResFormer pattern — alternating layers)
- Sliding window attention (configurable full/half context per layer)
- ReLU² activation in the MLP
- Softcap on logits (`15 × tanh(logits/15)`) to prevent extreme values

**The Optimizer (Muon + AdamW Hybrid):**
- **Muon** for 2D matrix parameters (uses orthogonalization via polar decomposition)
- **AdamW** for everything else (embeddings, scalars)
- Separate learning rates for embeddings, unembeddings, matrices, and scalars
- Dynamic scheduling: warmup → constant → cooldown over the 5-minute budget

**Key Hyperparameters the Agent Can Tune:**
```
DEPTH = 4                    # transformer layers
ASPECT_RATIO = 64            # model_dim = depth × 64
TOTAL_BATCH_SIZE = 65,536    # tokens per optimizer step
EMBEDDING_LR = 0.6           # token embedding learning rate
MATRIX_LR = 0.04             # Muon learning rate
WARMDOWN_RATIO = 0.5         # LR cooldown fraction
WEIGHT_DECAY = 0.2           # Muon weight decay
```

The agent can change **any of these** — or rewrite the architecture entirely. The only rule: training must complete within 5 minutes.

### `program.md` — The Research Director (Human Edits This)

This is the "skill" file that tells the AI agent how to behave. It defines:
- How to set up experiments (create branch, initialize results log)
- The experimentation loop (modify → train → evaluate → keep/discard)
- Constraints (never modify `prepare.py`, 5-minute budget is sacred)
- Success criteria (lower `val_bpb` wins; simpler code preferred)

Think of it as **programming your research org in English**.

---

## Deep Dive: The Nitty-Gritty of How It Actually Works

The tweet by @hooeem popularized autoresearch as "set it and forget it ML" — but under the hood, the engineering decisions are what make the whole thing tick. Let's peel back each layer.

### The Data Pipeline: Zero-Waste Document Packing

Most training pipelines waste tokens through padding. Autoresearch uses **BOS-aligned best-fit packing** — a bin-packing algorithm that achieves 100% token utilization:

1. Every row in the batch starts with a BOS (beginning of sequence) token
2. Documents are packed back-to-back using a **best-fit strategy**: for each remaining slot in a row, the system finds the largest document from a buffer of 1,000 pre-tokenized documents that fits entirely
3. If no document fits the remaining space, it crops the shortest available document to fill exactly
4. No padding tokens. Ever. Every single token in the batch is meaningful

The data comes from HuggingFace's `climbmix-400b-shuffle` dataset (parquet shards), with one shard (shard #6542) permanently pinned as the validation set — guaranteeing every experiment is evaluated on identical data.

### The Tokenizer: Small and Fast

Instead of GPT-2's 50K+ vocab, autoresearch uses a **tiny 8,192-token BPE vocabulary** trained with `rustbpe` (a Rust-based BPE implementation for speed). The split pattern follows GPT-4's regex style. Four special tokens are reserved (`<|reserved_0|>` through `<|reserved_3|>`), with `reserved_0` serving as the BOS token.

Why such a small vocab? It keeps the embedding tables small, which matters enormously when you're training models on a MacBook with limited memory. The model dim is only 256 (4 layers x 64 aspect ratio) in the default config — a 50K vocab would bloat the embedding layer relative to the transformer.

### The Model: Modern Tricks in a Tiny Package

The default model is deliberately small (~11.5M parameters) but packs in state-of-the-art architectural ideas:

**Rotary Position Embeddings (RoPE):** Instead of learned positional embeddings, the model uses sinusoidal rotary embeddings pre-computed for `sequence_len × 10` positions (20,480). These are applied to both Q and K tensors. RoPE naturally encodes relative position, which generalizes better than absolute position encodings.

**Value Residual Connections (ResFormer):** This is a clever one. On alternating layers, the model maintains a separate **value embedding** — a parallel embedding lookup that gets mixed into the attention values via a learned, input-dependent gate:

```
gate = 2 × sigmoid(ve_gate(x[:, :, :32]))
v = v + gate × value_embedding
```

The gate is initialized to zero, so `sigmoid(0) = 0.5`, scaled by 2 = 1.0 — a neutral starting point. This gives the model a "shortcut" from the input directly into the attention values, helping gradients flow more easily through deep networks.

**ReLU² Activation:** The MLP doesn't use standard ReLU or GELU. It uses `ReLU(x)²` — ReLU followed by squaring. This creates a smoother activation landscape while still being sparse (values below zero stay at zero). It's been shown to improve training stability in certain regimes.

**Softcap on Logits:** Before the final softmax, logits are clamped via `15 × tanh(logits / 15)`. This prevents any single logit from becoming extremely large, which stabilizes training — especially important when training at high learning rates with aggressive optimizers like Muon.

**Per-Layer Learned Scales:** Each layer has two learned scalar parameters:
- `resid_lambdas[i]`: scales the residual stream before entering the layer
- `x0_lambdas[i]`: scales a **skip connection from the initial embedding** (`x0`)

The forward pass looks like: `x = resid_lambda * x + x0_lambda * x0` before each transformer block. This x0 skip connection is reminiscent of DenseNet — it lets deeper layers directly access the original token representation, initialized at 0.1 so the model starts nearly like a standard residual network.

**Sliding Window Attention:** The `WINDOW_PATTERN` string (e.g., "SSSL") configures per-layer attention windows. "L" = full context (2048 tokens), "S" = half context (1024 tokens). The last layer always gets full attention regardless of the pattern. This saves compute in earlier layers while preserving global context where it matters most.

### The Optimizer: Muon — The Secret Weapon

The most exotic piece is the **Muon optimizer**, used specifically for 2D matrix parameters (attention projections, MLP weights). Here's what makes it different from Adam:

**Polar Express Orthogonalization:** After computing the standard Nesterov momentum update, Muon runs the gradient through a **polar decomposition approximation** using pre-computed polynomial coefficients (5 iterations by default). This projects the gradient onto the manifold of orthogonal matrices. The intuition: weight matrices should preserve information, and orthogonal updates help maintain this property.

```python
# The actual "polar express" iteration
for a, b, c in polar_express_coeffs:
    A = X.mT @ X
    B = b * A + c * (A @ A)
    X = a * X + X @ B
```

**NorMuon Variance Reduction:** After orthogonalization, Muon applies a variance normalization step using an exponential moving average of per-row (or per-column) squared norms. This stabilizes the update magnitude across dimensions.

**Cautious Weight Decay:** Instead of standard weight decay (shrink all weights uniformly), Muon applies weight decay only where the gradient and the parameter have the **same sign** — meaning it only decays weights that the gradient is already pushing in the same direction. This prevents weight decay from fighting the optimizer.

**Separate Learning Rate Scaling:** AdamW handles embeddings, unembeddings, and per-layer scalars with separate learning rates. All AdamW learning rates are scaled by `1/sqrt(model_dim / 768)` to adjust for model width. Muon's learning rate is additionally scaled by `sqrt(max(1, rows/cols))` to account for matrix aspect ratios.

**Dynamic Schedules Over 5 Minutes:**
- Learning rate: warmup (default 0%) → constant → cooldown over last 50% of budget
- Muon momentum: linearly interpolates from 0.85 → 0.95 over the first 300 steps
- Weight decay: linearly decays from 0.2 → 0 over training (less regularization as the model converges)

### The Training Loop: Time-Budgeted with Safety Rails

The training loop has several non-obvious design decisions:

**Wall-clock timing, not step-based:** Training runs for exactly 300 seconds of wall-clock time. The first 10 steps are excluded from timing (warmup for torch.compile and kernel compilation). This means faster hardware processes more tokens in the same budget — the model architecture that's optimal on an M4 Max may differ from what's optimal on an H100.

**Gradient accumulation:** The effective batch size is 65K tokens, but this is achieved via multiple micro-steps of `DEVICE_BATCH_SIZE=16` sequences of 2048 tokens each. On MPS with limited memory, this lets you train with a large effective batch size without OOM.

**Fast-fail on NaN/explosion:** If the training loss exceeds 100 at any point, the script immediately exits with status code 1. This lets the agent quickly move on from unstable hyperparameter configurations.

**GC management:** Python's garbage collector is frozen after the first step and only runs every 5,000 steps. GC pauses cause ~500ms stalls that would waste precious seconds of the 5-minute budget.

**Prefetching:** The next batch is loaded during the backward pass of the current step, overlapping data loading with computation.

### The Evaluation Metric: Bits Per Byte (BPB)

The choice of metric is critical. Val BPB measures cross-entropy loss in bits per byte of text, not per token. This makes it **vocabulary-size independent** — if the agent changes the tokenizer's effective vocabulary (via architecture changes that affect how tokens are weighted), the metric stays fair.

The evaluation:
1. Runs on exactly `40 × 524,288 = 20,971,520` tokens from the pinned validation shard
2. Computes per-token cross-entropy loss (unreduced)
3. Weights each token by its byte length (looked up from a pre-computed table)
4. Special tokens (byte length 0) are masked out from both the loss sum and byte count
5. Converts from nats to bits: `BPB = total_nats / (log(2) × total_bytes)`

### The Agent Loop: Git as an Experiment Tracker

This is where the autonomous research actually happens. The agent:

1. Creates a dedicated git branch (e.g., `autoresearch/mar5`)
2. Runs the baseline `train.py` to establish a starting `val_bpb`
3. Modifies `train.py` with an experimental idea
4. Commits the change, then runs `uv run train.py > run.log 2>&1`
5. Extracts `val_bpb` and `peak_vram_mb` from the log via grep
6. If improved → keeps the commit, logs "keep" to `results.tsv`
7. If worse → `git reset` back to the previous state, logs "discard"
8. If crashed → reads the traceback, optionally fixes trivial bugs, logs "crash"
9. **Goes back to step 3. Forever.** The agent is explicitly instructed to never stop.

The `results.tsv` becomes a complete experiment log:

```
commit    val_bpb     memory_gb   status    description
a1b2c3d   0.997900    44.0        keep      baseline
b2c3d4e   0.993200    44.2        keep      increase LR to 0.04
c3d4e5f   1.005000    44.0        discard   switch to GeLU activation
d4e5f6g   0.000000    0.0         crash     double model width (OOM)
```

Git history becomes your experiment tracker. Each successful experiment is a commit. You can `git log` to see the progression, `git diff` between commits to see exactly what changed, and `git bisect` to find which change mattered most.

---

## What Makes the macOS Fork Special?

Apple Silicon Macs use **Metal Performance Shaders (MPS)** instead of CUDA. This required several targeted adaptations:

| Feature | NVIDIA (CUDA) | Apple Silicon (MPS) |
|---|---|---|
| **Precision** | bfloat16 autocast | No autocast (MPS doesn't support it) |
| **Compilation** | torch.compile enabled | torch.compile disabled (unstable on Metal) |
| **Attention** | FlashAttention-3 | PyTorch SDPA with manual sliding window masks |
| **Memory** | Large batch sizes | Tuned batch sizes for Metal memory bounds |
| **Sync** | CUDA synchronize | MPS synchronize |

The fork also adds:
- Platform verification at startup (checks for macOS + MPS availability)
- Explicit optimizer state casting for MPS device/dtype compatibility
- Fallback paths throughout the codebase that preserve CUDA support

The result: **the same autonomous research loop running natively on your MacBook**.

---

## How to Set It Up (10 Minutes)

### Prerequisites
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- An AI coding assistant (Claude Code at $20/month, or free Cursor)

### Step 1: Install the Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Clone and Install
```bash
git clone https://github.com/miolini/autoresearch-macos.git
cd autoresearch-macos
uv sync
```

### Step 3: Prepare Data (One-Time, ~2 min)
```bash
uv run prepare.py
```
This downloads training data shards and trains a BPE tokenizer. Only needs to run once.

### Step 4: Verify with a Manual Run (~5 min)
```bash
uv run train.py
```
You should see training metrics streaming, followed by a final output like:
```
val_bpb:          0.997900
training_seconds: 300.1
total_tokens_M:   499.6
num_steps:        953
```

### Step 5: Launch the Agent
Open your AI coding assistant in the repo directory and prompt:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

Then walk away. The agent will:
- Create a branch (`autoresearch/<tag>`)
- Initialize a `results.tsv` log
- Start experimenting autonomously

---

## The Experiment Loop Visualized

```
┌─────────────────────────────────────────┐
│           program.md (Human)            │
│    "Here's how to do research"          │
└──────────────┬──────────────────────────┘
               │ reads
               ▼
┌─────────────────────────────────────────┐
│           AI Agent (Claude/Cursor)      │
│                                         │
│  1. Modify train.py                     │
│  2. Run: uv run train.py > run.log      │
│  3. Extract val_bpb from output         │
│  4. Improved? → git commit              │
│     Not improved? → git reset           │
│  5. Log result to results.tsv           │
│  6. Go to 1                             │
│                                         │
└──────────────┬──────────────────────────┘
               │ edits
               ▼
┌─────────────────────────────────────────┐
│           train.py                      │
│    Model + Optimizer + Training Loop    │
│    (architecture, hyperparams, etc.)    │
└──────────────┬──────────────────────────┘
               │ imports
               ▼
┌─────────────────────────────────────────┐
│           prepare.py (Fixed)            │
│    Data loading + Tokenizer + Eval      │
│    (never modified = fair comparison)   │
└─────────────────────────────────────────┘
```

---

## Why This Matters

This isn't just a cool hack. It's a glimpse at the future of ML research:

1. **Democratized experimentation.** You don't need an H100 cluster. A MacBook and an overnight run gives you 100 experiments.

2. **Research as programming.** Instead of manually tuning hyperparameters, you program `program.md` — the instructions that guide autonomous research. The meta-game becomes: *what instructions produce the fastest research progress?*

3. **Fair comparison by design.** Fixed time budget + fixed evaluation = every experiment is directly comparable, regardless of what the agent changed.

4. **Compound improvements.** Each successful experiment builds on the last. Over 100 iterations, small gains compound into meaningful improvements.

The era of "set it and forget it" ML research has begun — and it runs on your Mac.

---

*Inspired by Andrej Karpathy's autoresearch project and the macOS fork by miolini. Try it yourself: [github.com/miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)*

---

#MachineLearning #AI #AppleSilicon #LLM #DeepLearning #Autoresearch #MacOS #MLEngineering #ArtificialIntelligence #AndrejKarpathy
