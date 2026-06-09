# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Group theory-based Rubik's cube representation, visualization, and ML solver. Two solver approaches run in parallel:
1. **Transformer** — autoregressive next-move prediction (LLM-style tokenization)
2. **Discrete diffusion (D3PM)** — forward process = applying random moves, reverse = learning to undo them

Training starts with random scrambles; later a CFOP-restricted variant (OLL/PLL) will be added. Framework: MLX (Apple Silicon). Final target: Swift app with 3D visualization and embedded MLX model.

## Environment

Python 3.12, venv at `cubea/`.

```bash
source cubea/bin/activate
pip install -r requirements.txt   # torch, torchvision, torchaudio (→ will migrate to mlx)
```

macOS Apple Silicon. Current code uses PyTorch MPS; migration to MLX is planned.

## Running Code

No formal test runner. Each module has a `__main__` block for verification:

```bash
cd source/cube
python state.py      # group algebra checks (composition, inversion, multiplication)
python vis_util.py   # net rendering and move correctness assertions
```

## Repository Layout (Planned)

```
source/
  cube/
    state.py          # Group-theoretic State: core representation + move algebra
    vis_util.py       # State → 6×3×3 net tensor; ASCII net printer; (→ add 3D)
  model/
    transformer.py    # Autoregressive next-move predictor (LLM-style)
    diffusion.py      # Discrete diffusion solver (D3PM)
  train.py            # Training entry point (MLX)
  infer.py            # Inference entry point (MLX)
  playground.py       # Scratch / prototyping
  cube_template.html  # HTML/CSS net visualizer (static template)
swift/                # (planned) Swift app — SceneKit/RealityKit 3D + MLX Swift
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

Constants: `CORNER_FACES_POSITIONS`, `EDGE_FACES_POSITIONS` define which face cells each corner/edge piece occupies. `TWIST_SUBCUBE_MAP` and `FLIP_SUBCUBE_MAP` determine sticker permutation under twist/flip.

3D visualization (Python side) is planned — likely matplotlib or a dedicated 3D library.

### Diffusion Model (planned)

- **Forward process**: apply `t` random moves from `MOVES` to the solved state → noisy state at level `t`
- **Reverse process**: model predicts which move (from the 18-move vocabulary: 6 faces × {1, 2, 3 turns}) to apply to reduce noise by 1 step
- Loss: cross-entropy over the 18-class move vocabulary at each timestep
- Noise schedule: uniform random move selection (can later weight by move type)

### Transformer Model (planned)

- Tokenize: cube state (encoded as a fixed-length vector) + move history as token sequence
- Output: next-move token (18-class or 6-class + modifier)
- Autoregressive decoding at inference time

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
