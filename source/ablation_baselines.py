"""Deterministic (no-LLM) baselines for the LLM-cube ablation.

Two reference policies that bracket the LLM agents:
  - random : apply a random move each step (banning the immediate inverse) — the
             floor; shows how much a blind random walk solves by luck.
  - greedy : apply rank_moves[0] each step (pure value-net greedy) — the ceiling
             the intuition-equipped LLM could reach if it always took the top move.

Run at high N (cheap, no network) to anchor the figure.

    uv run python source/ablation_baselines.py --n 100 --depths 1-12
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cube_tools import CubeSession, NOTATION  # noqa: E402

try:
    from data import _INV_IDX  # noqa: E402
except Exception:  # noqa: BLE001
    _INV_IDX = None


def _budget(depth: int) -> int:
    return max(20, depth * 4)


def random_solve(seed: int, depth: int) -> bool:
    s = CubeSession(seed=seed)
    s.scramble(depth)
    rng = random.Random(seed * 7919 + 1)
    last = -1
    for _ in range(_budget(depth)):
        if s.observe()["is_solved"]:
            return True
        i = rng.randrange(18)
        if _INV_IDX is not None and last >= 0:
            while i == _INV_IDX[last]:
                i = rng.randrange(18)
        s.apply(NOTATION[i])
        last = i
    return s.observe()["is_solved"]


def greedy_solve(seed: int, depth: int) -> bool:
    """Perfect greedy on the learned value (the good heuristic) = the ceiling."""
    s = CubeSession(seed=seed)
    s.scramble(depth)
    for _ in range(_budget(depth)):
        if s.observe()["is_solved"]:
            return True
        best = s.rank_moves()["ranked_moves"][0]
        s.apply(best["move"])
    return s.observe()["is_solved"]


def greedy_pieces_solve(seed: int, depth: int) -> bool:
    """Perfect greedy on pieces_solved (the bad heuristic the blind agent has)."""
    s = CubeSession(seed=seed)
    s.scramble(depth)
    for _ in range(_budget(depth)):
        if s.observe()["is_solved"]:
            return True
        best = s.rank_moves_pieces()["ranked_moves"][0]
        s.apply(best["move"])
    return s.observe()["is_solved"]


def _parse_depths(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--depths", default="1-12")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    depths = _parse_depths(args.depths)
    logf = open(args.out, "a") if args.out else None

    def log(m: str) -> None:
        print(m, flush=True)
        if logf:
            logf.write(m + "\n"); logf.flush()

    log(f"=== deterministic baselines | n={args.n} depths={depths} ===")
    log(f"{'depth':>5} {'random':>8} {'gr_pieces':>10} {'gr_value':>9}")
    for d in depths:
        r = sum(random_solve(seed, d) for seed in range(args.n))
        gp = sum(greedy_pieces_solve(seed, d) for seed in range(args.n))
        gv = sum(greedy_solve(seed, d) for seed in range(args.n))
        log(f"{d:>5} {r/args.n:>8.2f} {gp/args.n:>10.2f} {gv/args.n:>9.2f}")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
