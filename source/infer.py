"""Evaluation harness for trained CubeSolver models.

Runs a trained model on a batch of scrambled cubes via greedy rollout and
measures how often it actually reaches the solved state.

Usage
-----
    uv run python source/infer.py --ckpt runs/bc/latest.npz --baseline --n 200
    uv run python source/infer.py \\
        --ckpt runs/bc/latest.npz \\
        --ckpt runs/diffusion/latest.npz \\
        --baseline --n 200 --scramble-depth 25
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cube"))

from data import _IDENTITY, _MOVES_PY, _compose   # noqa: E402
from cfop import solve as cfop_solve, cube_solved  # noqa: E402
from model.solver import CubeSolver                # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path: str,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ffn_mult: int = 4,
    t_max: int = 100,
) -> CubeSolver:
    """Build a CubeSolver with the given config and load weights from ckpt_path."""
    model = CubeSolver(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_mult=ffn_mult,
        t_max=t_max,
    )
    weights = dict(mx.load(ckpt_path).items())
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def states_to_arrays(
    states: list[tuple],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Convert a list of N state tuples to four int32 MLX arrays.

    Returns (cp, ct, ep, ef) with shapes [N,8], [N,8], [N,12], [N,12].
    """
    cp_rows, ct_rows, ep_rows, ef_rows = [], [], [], []
    for cp, ct, ep, ef in states:
        cp_rows.append(cp)
        ct_rows.append(ct)
        ep_rows.append(ep)
        ef_rows.append(ef)
    return (
        mx.array(cp_rows, dtype=mx.int32),
        mx.array(ct_rows, dtype=mx.int32),
        mx.array(ep_rows, dtype=mx.int32),
        mx.array(ef_rows, dtype=mx.int32),
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(
    model: CubeSolver,
    scrambles: list[tuple],
    scramble_depth: int,
    max_steps: int,
) -> tuple[list[bool], list[int]]:
    """Greedy model rollout for a batch of scrambled cubes.

    Parameters
    ----------
    model         : loaded CubeSolver
    scrambles     : list of N scrambled state tuples (cp,ct,ep,ef)
    scramble_depth: used to set the initial t value (noise level counts down)
    max_steps     : maximum number of steps to attempt

    Returns
    -------
    solved_mask : list[bool] of length N, True if the cube was solved
    steps       : list[int], solved_step (1-indexed) or max_steps if unsolved
    """
    n = len(scrambles)
    current_states = [s for s in scrambles]  # mutable copy
    solved_step = [-1] * n                   # -1 = unsolved so far

    # Goal for all cubes is the identity (solved state)
    goal_cp, goal_ct, goal_ep, goal_ef = states_to_arrays([_IDENTITY] * n)

    for step in range(max_steps):
        # Only process cubes not yet solved
        active = [i for i in range(n) if solved_step[i] == -1]
        if not active:
            break

        # Build arrays for the full batch (inactive cubes still get a prediction
        # but we ignore it; this keeps array shapes uniform and avoids reindexing)
        curr_cp, curr_ct, curr_ep, curr_ef = states_to_arrays(current_states)

        # Noise level counts down: starts at scramble_depth, decreases each step
        t_val = max(1, scramble_depth - step)
        t = mx.array([t_val] * n, dtype=mx.int32)

        logits = model(
            goal=(goal_cp, goal_ct, goal_ep, goal_ef),
            curr=(curr_cp, curr_ct, curr_ep, curr_ef),
            t=t,
        )
        mx.eval(logits)
        preds = mx.argmax(logits, axis=1).tolist()

        # Apply predicted move to each not-yet-solved cube
        for i in active:
            move_idx = int(preds[i])
            current_states[i] = _compose(current_states[i], _MOVES_PY[move_idx])
            if cube_solved(current_states[i]):
                solved_step[i] = step + 1  # 1-indexed step count

    solved_mask = [solved_step[i] != -1 for i in range(n)]
    steps = [solved_step[i] if solved_step[i] != -1 else max_steps for i in range(n)]
    return solved_mask, steps


# ---------------------------------------------------------------------------
# Scramble generation (fixed-seed for reproducibility)
# ---------------------------------------------------------------------------

def _generate_scrambles(n: int, scramble_depth: int, seed: int = 0) -> list[tuple]:
    """Generate n scrambled cubes from the identity using a fixed random seed."""
    rng = random.Random(seed)
    scrambles = []
    for _ in range(n):
        state = _IDENTITY
        for _ in range(scramble_depth):
            state = _compose(state, _MOVES_PY[rng.randrange(18)])
        scrambles.append(state)
    return scrambles


# ---------------------------------------------------------------------------
# Evaluate a single checkpoint
# ---------------------------------------------------------------------------

def evaluate(
    ckpt_path: str,
    n: int = 200,
    scramble_depth: int = 25,
    max_steps: int | None = None,
    seed: int = 0,
    **model_cfg,
) -> dict:
    """Evaluate a checkpoint on n scrambles.

    Parameters
    ----------
    ckpt_path     : path to the .npz checkpoint
    n             : number of scrambles to evaluate
    scramble_depth: depth of each scramble
    max_steps     : maximum rollout steps (default: scramble_depth * 4)
    seed          : random seed for scramble generation
    **model_cfg   : forwarded to load_model (d_model, n_layers, etc.)

    Returns
    -------
    dict with keys: success_rate, n_solved, n, avg_steps_solved,
                    median_steps_solved, scramble_depth, max_steps
    """
    if max_steps is None or max_steps <= 0:
        max_steps = scramble_depth * 4

    model = load_model(ckpt_path, **model_cfg)
    scrambles = _generate_scrambles(n, scramble_depth, seed=seed)

    solved_mask, steps = rollout(model, scrambles, scramble_depth, max_steps)

    n_solved = sum(solved_mask)
    success_rate = n_solved / n

    solved_steps = [steps[i] for i in range(n) if solved_mask[i]]
    avg_steps = statistics.mean(solved_steps) if solved_steps else float("nan")
    median_steps = statistics.median(solved_steps) if solved_steps else float("nan")

    return {
        "success_rate": success_rate,
        "n_solved": n_solved,
        "n": n,
        "avg_steps_solved": avg_steps,
        "median_steps_solved": median_steps,
        "scramble_depth": scramble_depth,
        "max_steps": max_steps,
    }


# ---------------------------------------------------------------------------
# CFOP baseline
# ---------------------------------------------------------------------------

def cfop_baseline(n: int, scramble_depth: int, seed: int = 0) -> dict:
    """Run the CFOP solver on the same n scrambles and report solution lengths.

    Returns dict with keys: success_rate, n_solved, n, avg_len, median_len,
                            scramble_depth
    """
    scrambles = _generate_scrambles(n, scramble_depth, seed=seed)
    lengths = []
    n_solved = 0
    for state in scrambles:
        try:
            sol = cfop_solve(state)
            if cube_solved(_compose_seq(state, sol)):
                lengths.append(len(sol))
                n_solved += 1
        except RuntimeError:
            pass

    avg_len = statistics.mean(lengths) if lengths else float("nan")
    median_len = statistics.median(lengths) if lengths else float("nan")

    return {
        "success_rate": n_solved / n,
        "n_solved": n_solved,
        "n": n,
        "avg_len": avg_len,
        "median_len": median_len,
        "scramble_depth": scramble_depth,
    }


def _compose_seq(state: tuple, moves: list[int]) -> tuple:
    """Apply a sequence of move indices to a state."""
    for mi in moves:
        state = _compose(state, _MOVES_PY[mi])
    return state


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    """Print a comparison table of evaluation results."""
    col_w = [40, 14, 12, 12]
    header = (
        f"{'label':<{col_w[0]}}"
        f"{'success_rate':>{col_w[1]}}"
        f"{'avg_steps':>{col_w[2]}}"
        f"{'median':>{col_w[3]}}"
    )
    sep = "-" * sum(col_w)
    print()
    print(header)
    print(sep)
    for row in rows:
        label = row["label"]
        sr = row.get("success_rate", float("nan"))
        avg = row.get("avg_steps", row.get("avg_len", float("nan")))
        med = row.get("median_steps", row.get("median_len", float("nan")))

        avg_s = f"{avg:.1f}" if avg == avg else "n/a"
        med_s = f"{med:.1f}" if med == med else "n/a"

        print(
            f"{label:<{col_w[0]}}"
            f"{sr:>{col_w[1]}.3f}"
            f"{avg_s:>{col_w[2]}}"
            f"{med_s:>{col_w[3]}}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained CubeSolver checkpoints."
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        dest="ckpts",
        metavar="PATH",
        help="Path to a checkpoint .npz (can be given multiple times).",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of scrambles.")
    parser.add_argument(
        "--scramble-depth", type=int, default=25, help="Random moves per scramble."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max rollout steps (0 = auto: scramble_depth * 4).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run the CFOP baseline on the same scrambles.",
    )
    # Model architecture args (for non-default checkpoints)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--t-max", type=int, default=100)

    args = parser.parse_args()

    max_steps = args.max_steps if args.max_steps > 0 else None
    effective_max = max_steps if max_steps is not None else args.scramble_depth * 4

    model_cfg = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        t_max=args.t_max,
    )

    print(
        f"Evaluating: n={args.n}, scramble_depth={args.scramble_depth}, "
        f"max_steps={effective_max}, seed={args.seed}"
    )

    rows = []

    if args.ckpts:
        for ckpt_path in args.ckpts:
            print(f"  loading {ckpt_path} ...", flush=True)
            result = evaluate(
                ckpt_path,
                n=args.n,
                scramble_depth=args.scramble_depth,
                max_steps=max_steps,
                seed=args.seed,
                **model_cfg,
            )
            rows.append(
                {
                    "label": ckpt_path,
                    "success_rate": result["success_rate"],
                    "avg_steps": result["avg_steps_solved"],
                    "median_steps": result["median_steps_solved"],
                }
            )
            print(
                f"    solved {result['n_solved']}/{result['n']} "
                f"({result['success_rate']:.1%})",
                flush=True,
            )

    if args.baseline:
        print("  running CFOP baseline ...", flush=True)
        base = cfop_baseline(args.n, args.scramble_depth, seed=args.seed)
        rows.append(
            {
                "label": "CFOP baseline",
                "success_rate": base["success_rate"],
                "avg_steps": base["avg_len"],
                "median_steps": base["median_len"],
            }
        )
        print(
            f"    solved {base['n_solved']}/{base['n']} "
            f"({base['success_rate']:.1%})",
            flush=True,
        )

    _print_table(rows)


if __name__ == "__main__":
    main()
