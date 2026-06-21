# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Group theory-based Rubik's cube representation and visualization, used as an
**instrument** for several parallel solver tracks. The cube engine is the shared
foundation; the solvers are research probes on top of it.

1. **Group-theoretic core + visualization** — the `State` class and net renderer (done, oracle-tested)
2. **Classical CFOP solver** — Cross→F2L→OLL→PLL, mainly for behavioral-cloning data (done)
3. **Deep-learning solvers** — Transformer (BC) and discrete diffusion (D3PM), plus value iteration; experimental, results recorded in `docs/EXPERIMENTS.md`, currently parked
4. **LLM pseudo-intuition agent** — cube ops exposed as function tools; a small local LLM (Ollama qwen3.5:4b/9b) solves the cube. Ablation isolates "tools vs lookahead vs intuition." Written up in `docs/article_llm_intuition.md`
5. **Swift 3D app** — SceneKit/RealityKit + embedded MLX Swift (planned)

Framework: MLX (Apple Silicon). Final target: the Swift app with an embedded model.

## Environment

Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/). Dependencies are
declared in `pyproject.toml` (runtime: `mlx`; dev: `pytest`, `pytest-xdist`).

```bash
uv sync                       # create .venv and install deps + dev group
```

macOS Apple Silicon. Backend is **MLX** (the old PyTorch/MPS code has been
fully migrated). No GitHub Actions yet — a self-hosted runner is planned, so do
not add GHA workflows.

## Running Code

Tests use `pytest` (validated against an independent facelet oracle):

```bash
uv run pytest            # full suite
uv run pytest -n auto    # parallel (pytest-xdist)
```

Each module also has a `__main__` block for quick manual checks:

```bash
cd source/cube
uv run python state.py      # identity / inverse / order-4 checks for all moves
uv run python vis_util.py   # prints the net for solved + all 6 quarter turns
```

## Repository Layout

```
source/
  cube/
    state.py             # Group-theoretic State: core representation + move algebra
    vis_util.py          # State → 6×3×3 net tensor; ASCII net printer; (→ add 3D)
  model/
    transformer.py       # Autoregressive next-move predictor (LLM-style)
    diffusion.py         # Discrete diffusion solver (D3PM)
  cfop.py                # Classical CFOP solver (Cross/F2L BFS + OLL/PLL tables)
  data.py                # Scramble/dataset generation + move-index helpers
  train.py               # DL training entry point (MLX)
  infer.py               # DL evaluation harness (greedy/beam rollout success)
  cube_tools.py          # LLM tool kernel: observe/simulate/apply/rank_moves/memory
  llm_agent.py           # Ollama-driven agent harness (blind/intuition/pieces/memory)
  ablation_baselines.py  # Deterministic floor/ceiling baselines (no LLM)
  playground.py          # Scratch / prototyping
  cube_template.html     # HTML/CSS net visualizer (static template)
docs/
  EXPERIMENTS.md         # DL-track experiment log + Fable5 review + negative results
  article_llm_intuition.md  # LLM pseudo-intuition writeup (Qiita-bound)
swift/                   # (planned) Swift app — SceneKit/RealityKit 3D + MLX Swift
```

## Core Architecture

### State (`state.py`)

`State` is both a cube state and a move — the same class represents group elements.

Tensors:
- `corner_positions`: `int8[8]` — which corner piece is at each slot
- `corner_orientations`: `float16[8, 2]` — `[cos θ, sin θ]` of twist (complex multiplication for composition)
- `edge_positions`: `int8[12]` — which edge piece is at each slot
- `edge_orientations`: `int8[12]` — flip as `+1` / `-1`

Derived integer views (for equality checks, diffusion labels):
- `state.twist_co` → `int8[8]` twist index 0/1/2 via nearest-vector lookup into `TWIST_TABLE`
- `state.twist_eo` → `int8[12]` flip 0/1

Group operators:
| Op | Semantics |
|---|---|
| `a @ b` | Immutable compose: apply `b` to `a` |
| `a @= b` | Mutable compose |
| `~a` | Inverse element |
| `n * a` | Repeat `n` times (negative = inverse direction) |
| `a == b` | Equality via integer twist/flip (not raw float) |
| `State()` | Identity (solved cube) |

`MOVES` dict: `'U'`, `'D'`, `'L'`, `'R'`, `'F'`, `'B'` — all 6 quarter-turn `State` objects.

**Mutation safety**: `apply()` mutates in place; `get_applied()` / `@` do not. Use `.clone()` before mutating when the original must be preserved.

### Visualization (`vis_util.py`)

`state_to_net(state)` → `int64[6, 3, 3]` tensor — each cell is a face index 0–5 (F=0, R=1, L=2, B=3, U=4, D=5).

`print_net(net)` renders the unfolded cross layout:
```
      U
  L F R B
      D
```

Constants `CORNER_FACES_POSITIONS` / `EDGE_FACES_POSITIONS` list, for each slot, the `(face, cell)` of every sticker in a **canonical CCW-from-outside order**. Because that order has uniform chirality across all slots, twist/flip render with a single cyclic-offset formula (no per-piece special-casing):

```
corner: net[slot.faces[(k + twist) % 3]] = piece.colors[k]
edge:   net[slot.faces[(k + flip)  % 2]] = piece.colors[k]
```

`tests/oracle.py` is an independent 3D facelet model (rotates the 54 stickers directly); `tests/test_vis_util.py` checks `state_to_net` against it for all moves and random scrambles. The `MOVES` tables and these canonical orders were derived from that geometry.

3D visualization (Python side) is planned — likely matplotlib or a dedicated 3D library.

### Diffusion Model (implemented, parked)

- **Forward process**: apply `t` random moves from `MOVES` to the solved state → noisy state at level `t`
- **Reverse process**: model predicts which move (from the 18-move vocabulary: 6 faces × {1, 2, 3 turns}) to apply to reduce noise by 1 step
- Loss: cross-entropy over the 18-class move vocabulary at each timestep
- Noise schedule: uniform random move selection (can later weight by move type)

### Transformer Model (implemented, parked)

- Tokenize: cube state (encoded as a fixed-length vector) + move history as token sequence
- Output: next-move token (18-class or 6-class + modifier)
- Autoregressive decoding at inference time

See `docs/EXPERIMENTS.md` for results: diffusion beats BC in rollout but plateaus,
goal-conditioning was never trained, and value iteration (DeepCubeA/DAVI) regressed
(recorded as a negative result).

### LLM Pseudo-Intuition Agent (`cube_tools.py`, `llm_agent.py`)

A small local LLM (Ollama qwen3.5:4b/9b) drives an observe→tool→observe loop. The
cube state is ground truth held by `CubeSession` (the LLM never tracks it mentally).

- `cube_tools.py` — tool kernel: `observe` (color net + per-face progress), `simulate`
  (non-committing preview), `apply`, `inverse`, `rank_moves` (18-move value ranking =
  the "intuition"), `rank_moves_pieces` (a deliberately bad heuristic for ablation),
  and a JSON-persisted `MacroMemory` (save/list/get/apply discovered algorithms)
- `llm_agent.py` — harness with modes `blind` / `intuition` / `pieces` / `memory`;
  instruments **rank-1 adherence** (言うとおり率: how often the model plays the
  intuition's top move)
- `ablation_baselines.py` — deterministic floor (random) and ceiling (perfect greedy
  on value vs on pieces) to bracket the LLM

The intuition is a separately trained ~0.8M-param value function (goal-conditioned,
regresses moves-to-solve), loaded from `runs/hindsight_not/latest.npz`. Core finding:
tools alone don't solve it, lookahead alone doesn't either; **heuristic quality
(intuition) is decisive**, adherence gates performance and scales with model size,
and self-improvement (memory) was NULL at 4b. The natural next step (article §6) is a
plan→execute→reflect loop. See `docs/article_llm_intuition.md`.

## Key Numbers

| Concept | Value |
|---|---|
| Corner slots | 8 |
| Edge slots | 12 |
| Face count | 6 (F, R, L, B, U, D) |
| Quarter-turn moves | 6 |
| Full move vocabulary (QTM) | 18 (6 faces × 1/2/3 turns) |
| Corner twist values | 0, 1, 2 (mod 3) |
| Edge flip values | 0, 1 (mod 2) |
