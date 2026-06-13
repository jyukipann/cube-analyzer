"""Evaluation harness for trained CubeSolver models.

Runs a trained model on a batch of scrambled cubes via greedy rollout or
beam-search rollout, and measures how often it actually reaches the solved
state.

Usage
-----
    uv run python source/infer.py --ckpt runs/bc/latest.npz --baseline --n 200
    uv run python source/infer.py \\
        --ckpt runs/bc/latest.npz \\
        --ckpt runs/diffusion/latest.npz \\
        --baseline --n 200 --scramble-depth 25
    uv run python source/infer.py \\
        --ckpt runs/diffusion/latest.npz \\
        --n 100 --scramble-depth 8 --beam 8
    uv run python source/infer.py \\
        --ckpt runs/diffusion/latest.npz \\
        --n 100 --scramble-depth 8 --t-mode none
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cube"))

from data import _IDENTITY, _MOVES_PY, _compose   # noqa: E402
from cfop import solve as cfop_solve, cube_solved, _INV_IDX, _state_key  # noqa: E402
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


def load_model_auto(
    ckpt_path: str,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ffn_mult: int = 4,
    t_max: int = 100,
) -> CubeSolver:
    """Load a CubeSolver, auto-reading config from a sibling .json if present.

    Looks for <ckpt_path>.json (i.e. latest.npz -> latest.json, or
    ckpt_5000.npz -> ckpt_5000.json). If found, overrides the supplied
    CLI defaults with the saved values, so the caller does not need to
    specify --d-model etc. manually.
    """
    cfg_path = Path(str(ckpt_path)).with_suffix(".json")
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            d_model  = cfg.get("d_model",  d_model)
            n_layers = cfg.get("n_layers", n_layers)
            n_heads  = cfg.get("n_heads",  n_heads)
            ffn_mult = cfg.get("ffn_mult", ffn_mult)
            t_max    = cfg.get("t_max",    t_max)
            print(f"    config loaded from {cfg_path.name}: "
                  f"d={d_model}, layers={n_layers}, heads={n_heads}, "
                  f"ffn_mult={ffn_mult}, t_max={t_max}")
        except Exception as exc:
            print(f"    warning: could not read {cfg_path}: {exc}; using defaults")

    return load_model(
        ckpt_path,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_mult=ffn_mult,
        t_max=t_max,
    )


# ---------------------------------------------------------------------------
# t-conditioning helpers
# ---------------------------------------------------------------------------

def _t_value(t_mode: str, scramble_depth: int, step: int, t_const: int) -> int:
    """Return the scalar t value for the given step under the chosen mode."""
    if t_mode == "countdown":
        return max(1, scramble_depth - step)
    elif t_mode == "const":
        return t_const
    elif t_mode == "none":
        return None  # caller must pass t=None to the model
    else:
        raise ValueError(f"Unknown t_mode: {t_mode!r}")


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
    t_mode: str = "countdown",
    t_const: int | None = None,
) -> tuple[list[bool], list[int]]:
    """Greedy model rollout for a batch of scrambled cubes.

    Parameters
    ----------
    model         : loaded CubeSolver
    scrambles     : list of N scrambled state tuples (cp,ct,ep,ef)
    scramble_depth: used to set the initial t value (noise level counts down)
    max_steps     : maximum number of steps to attempt
    t_mode        : 'countdown' | 'const' | 'none'
    t_const       : fixed t value used when t_mode == 'const'

    Returns
    -------
    solved_mask : list[bool] of length N, True if the cube was solved
    steps       : list[int], solved_step (1-indexed) or max_steps if unsolved
    """
    if t_const is None:
        t_const = scramble_depth

    n = len(scrambles)
    current_states = [s for s in scrambles]  # mutable copy
    solved_step = [-1] * n                   # -1 = unsolved so far
    prev_move = [-1] * n                     # last move applied, for inverse-ban
    visited: list[set] = [set() for _ in range(n)]  # cycle detection per cube

    # Seed visited sets with the initial (scrambled) states
    for i, s in enumerate(scrambles):
        visited[i].add(_state_key(s))

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

        # Compute t value for this step
        t_val = _t_value(t_mode, scramble_depth, step, t_const)
        if t_val is not None:
            t = mx.array([t_val] * n, dtype=mx.int32)
        else:
            t = None

        logits = model(
            goal=(goal_cp, goal_ct, goal_ep, goal_ef),
            curr=(curr_cp, curr_ct, curr_ep, curr_ef),
            t=t,
        )
        mx.eval(logits)
        logits_list = logits.tolist()  # [[18 floats], ...]

        # Apply predicted move to each not-yet-solved cube
        for i in active:
            row = logits_list[i]
            # Mask the immediate inverse of the previous move to -inf
            if prev_move[i] >= 0:
                inv_idx = _INV_IDX[prev_move[i]]
                row[inv_idx] = float("-inf")
            move_idx = int(max(range(18), key=lambda m: row[m]))

            next_state = _compose(current_states[i], _MOVES_PY[move_idx])
            key = _state_key(next_state)

            # Cycle detection: if this state was seen before, declare stuck
            if key in visited[i]:
                # Mark as stuck by flagging solved_step to a sentinel that
                # won't be -1 but also isn't a valid step; we use max_steps+1
                # to indicate "stuck, not solved"
                solved_step[i] = -(max_steps + 1)  # stuck sentinel
                continue

            visited[i].add(key)
            prev_move[i] = move_idx
            current_states[i] = next_state

            if cube_solved(current_states[i]):
                solved_step[i] = step + 1  # 1-indexed step count

    solved_mask = [solved_step[i] > 0 for i in range(n)]
    steps = [solved_step[i] if solved_step[i] > 0 else max_steps for i in range(n)]
    return solved_mask, steps


# ---------------------------------------------------------------------------
# Beam-search rollout
# ---------------------------------------------------------------------------

def rollout_beam(
    model: CubeSolver,
    scrambles: list[tuple],
    scramble_depth: int,
    max_steps: int,
    beam_width: int,
    t_mode: str = "countdown",
    t_const: int | None = None,
) -> tuple[list[bool], list[int]]:
    """Beam-search model rollout for a batch of scrambled cubes.

    Parameters
    ----------
    model         : loaded CubeSolver
    scrambles     : list of N scrambled state tuples (cp,ct,ep,ef)
    scramble_depth: used to set the initial t value (noise level counts down)
    max_steps     : maximum number of steps to attempt
    beam_width    : number of candidates to keep per scramble per step
    t_mode        : 'countdown' | 'const' | 'none'
    t_const       : fixed t value used when t_mode == 'const'

    Returns
    -------
    solved_mask : list[bool] of length N, True if the cube was solved
    steps       : list[int], solved_step (1-indexed) or max_steps if unsolved

    Notes
    -----
    Each scramble maintains an independent beam of up to `beam_width`
    (state, cumulative_logprob, last_move_idx) candidates. At each step we
    batch ALL live candidates across ALL unsolved scrambles into a single
    model forward pass, convert logits to log-probabilities, expand by the
    top-`beam_width` moves (banning the immediate inverse), deduplicate
    candidates by state within each beam (keeping the higher-logprob copy),
    and keep the top-`beam_width` children per scramble ranked by cumulative
    log-probability. If any candidate is solved we record that scramble as
    done and stop expanding it.
    """
    if t_const is None:
        t_const = scramble_depth

    n = len(scrambles)

    # Each scramble starts with a beam of one candidate:
    # (state, logprob=0.0, last_move_idx=-1)
    # beams[i] = list of (state_tuple, cumulative_logprob, last_move_idx)
    beams: list[list[tuple[tuple, float, int]]] = [
        [(s, 0.0, -1)] for s in scrambles
    ]

    solved_step = [-1] * n  # -1 = unsolved so far

    # Goal arrays are fixed (all identity) — we build once and reindex per batch
    goal_state = _IDENTITY

    for step in range(max_steps):
        # Indices of scrambles still being searched
        active = [i for i in range(n) if solved_step[i] == -1]
        if not active:
            break

        # Build the flat batch of ALL live candidates across active scrambles.
        # candidate_owner[k] = which scramble index owns the k-th candidate.
        candidate_states: list[tuple] = []
        candidate_owner: list[int] = []
        candidate_last_move: list[int] = []
        for i in active:
            for state, _lp, last_mi in beams[i]:
                candidate_states.append(state)
                candidate_owner.append(i)
                candidate_last_move.append(last_mi)

        batch_size = len(candidate_states)

        # Build MLX arrays for the whole batch
        curr_cp, curr_ct, curr_ep, curr_ef = states_to_arrays(candidate_states)
        goal_cp, goal_ct, goal_ep, goal_ef = states_to_arrays(
            [goal_state] * batch_size
        )

        t_val = _t_value(t_mode, scramble_depth, step, t_const)
        if t_val is not None:
            t = mx.array([t_val] * batch_size, dtype=mx.int32)
        else:
            t = None

        logits = model(
            goal=(goal_cp, goal_ct, goal_ep, goal_ef),
            curr=(curr_cp, curr_ct, curr_ep, curr_ef),
            t=t,
        )
        mx.eval(logits)

        # Convert logits -> log-probabilities via logsumexp (numerically stable)
        # logsumexp per row: shape [batch_size]
        log_z = mx.log(mx.sum(mx.exp(logits - mx.max(logits, axis=1, keepdims=True)), axis=1, keepdims=True)) + mx.max(logits, axis=1, keepdims=True)
        log_probs = logits - log_z          # shape [batch_size, 18]
        mx.eval(log_probs)
        log_probs_list = log_probs.tolist()  # list of list[float] len batch_size x 18

        # Map each candidate row back to (scramble_index, candidate_index_in_beam)
        # We need to know the parent's cumulative logprob; rebuild a lookup.
        # candidate_lp[k] = cumulative logprob of the k-th candidate in the batch
        candidate_lp: list[float] = []
        for i in active:
            for _state, lp, _mi in beams[i]:
                candidate_lp.append(lp)

        # Expand: for each scramble collect all (child_state, child_logprob, move_idx)
        # pairs, deduplicate by state key (keep highest logprob copy), then keep top
        # beam_width by child_logprob.
        # Group rows by scramble index.
        scramble_rows: dict[int, list[int]] = {i: [] for i in active}
        for k, i in enumerate(candidate_owner):
            scramble_rows[i].append(k)

        new_beams: dict[int, list[tuple[tuple, float, int]]] = {}
        for i in active:
            rows = scramble_rows[i]
            # Collect up to beam_width expansions per parent row, then take
            # global top beam_width across all parents for this scramble.
            children: list[tuple[float, tuple, int]] = []  # (logprob, state, move_idx)
            for k in rows:
                parent_lp = candidate_lp[k]
                parent_state = candidate_states[k]
                last_mi = candidate_last_move[k]
                row_log_probs = list(log_probs_list[k])  # 18 floats (copy)

                # Ban the immediate inverse of the last move
                if last_mi >= 0:
                    inv_idx = _INV_IDX[last_mi]
                    row_log_probs[inv_idx] = float("-inf")

                # Sort moves by logprob descending; only expand top beam_width
                sorted_moves = sorted(
                    range(18), key=lambda m: row_log_probs[m], reverse=True
                )[:beam_width]

                for move_idx in sorted_moves:
                    if row_log_probs[move_idx] == float("-inf"):
                        continue
                    child_state = _compose(parent_state, _MOVES_PY[move_idx])
                    child_lp = parent_lp + row_log_probs[move_idx]
                    children.append((child_lp, child_state, move_idx))

            # Deduplicate by state key: keep the copy with the highest logprob
            seen_keys: dict[tuple, int] = {}  # key -> index in deduped list
            deduped: list[tuple[float, tuple, int]] = []
            for child_lp, child_state, move_idx in children:
                k = _state_key(child_state)
                if k in seen_keys:
                    idx = seen_keys[k]
                    if child_lp > deduped[idx][0]:
                        deduped[idx] = (child_lp, child_state, move_idx)
                else:
                    seen_keys[k] = len(deduped)
                    deduped.append((child_lp, child_state, move_idx))

            # Keep top beam_width children; break ties arbitrarily
            deduped.sort(key=lambda x: x[0], reverse=True)
            deduped = deduped[:beam_width]

            new_beams[i] = [(st, lp, mi) for lp, st, mi in deduped]

        # Check for solved states; commit new beams
        for i in active:
            for state, lp, mi in new_beams[i]:
                if cube_solved(state):
                    solved_step[i] = step + 1  # 1-indexed
                    break
            beams[i] = new_beams[i]

    solved_mask = [solved_step[i] != -1 for i in range(n)]
    steps = [solved_step[i] if solved_step[i] != -1 else max_steps for i in range(n)]
    return solved_mask, steps


# ---------------------------------------------------------------------------
# Value-guided beam search rollout
# ---------------------------------------------------------------------------

def rollout_value_beam(
    model: CubeSolver,
    scrambles: list[tuple],
    scramble_depth: int,
    max_steps: int,
    beam_width: int,
    t_mode: str = "countdown",
    t_const: int | None = None,
) -> tuple[list[bool], list[int]]:
    """Policy-proposal + value-ranking beam search.

    At each step:
    1. Expand each live candidate with its policy's top-beam_width moves.
    2. Score every child by the value head (predicted cost-to-go).
    3. Keep the top-beam_width children per scramble with the LOWEST value.

    Both policy logits and value estimates come from the same batched forward
    pass (return_value=True).  A second batched forward scores the children.

    Parameters
    ----------
    model         : loaded CubeSolver (value head should be trained for best
                    results; untrained value head gracefully degrades to near-
                    random child ranking)
    scrambles     : list of N scrambled state tuples (cp,ct,ep,ef)
    scramble_depth: used to set the initial t value
    max_steps     : maximum rollout steps
    beam_width    : candidates to keep per scramble per step
    t_mode        : 'countdown' | 'const' | 'none'
    t_const       : fixed t for t_mode='const'

    Returns
    -------
    solved_mask : list[bool]
    steps       : list[int]
    """
    if t_const is None:
        t_const = scramble_depth

    n = len(scrambles)
    # beams[i] = list of (state_tuple, last_move_idx)
    beams: list[list[tuple[tuple, int]]] = [
        [(s, -1)] for s in scrambles
    ]
    solved_step = [-1] * n
    goal_state = _IDENTITY

    for step in range(max_steps):
        active = [i for i in range(n) if solved_step[i] == -1]
        if not active:
            break

        # ------------------------------------------------------------------
        # Step 1: collect all live candidates and run policy forward to get
        # top-beam_width child proposals per candidate.
        # ------------------------------------------------------------------
        candidate_states: list[tuple] = []
        candidate_owner: list[int] = []
        candidate_last_move: list[int] = []
        for i in active:
            for state, last_mi in beams[i]:
                candidate_states.append(state)
                candidate_owner.append(i)
                candidate_last_move.append(last_mi)

        batch_size = len(candidate_states)
        curr_cp, curr_ct, curr_ep, curr_ef = states_to_arrays(candidate_states)
        goal_cp, goal_ct, goal_ep, goal_ef = states_to_arrays(
            [goal_state] * batch_size
        )

        t_val = _t_value(t_mode, scramble_depth, step, t_const)
        t = mx.array([t_val] * batch_size, dtype=mx.int32) if t_val is not None else None

        # Policy forward (no value needed here; we only need logits for proposals)
        logits = model(
            goal=(goal_cp, goal_ct, goal_ep, goal_ef),
            curr=(curr_cp, curr_ct, curr_ep, curr_ef),
            t=t,
        )
        mx.eval(logits)
        logits_list = logits.tolist()

        # Expand: build children grouped by scramble
        scramble_children: dict[int, list[tuple[tuple, int]]] = {i: [] for i in active}
        for k, i in enumerate(candidate_owner):
            row = list(logits_list[k])
            last_mi = candidate_last_move[k]
            parent_state = candidate_states[k]

            # Ban the immediate inverse move
            if last_mi >= 0:
                row[_INV_IDX[last_mi]] = float("-inf")

            # Top beam_width proposals
            sorted_moves = sorted(range(18), key=lambda m: row[m], reverse=True)[:beam_width]
            for move_idx in sorted_moves:
                if row[move_idx] == float("-inf"):
                    continue
                child_state = _compose(parent_state, _MOVES_PY[move_idx])
                scramble_children[i].append((child_state, move_idx))

        # ------------------------------------------------------------------
        # Step 2: deduplicate children by state key per scramble, then score
        # all children with the value head in one batched forward.
        # ------------------------------------------------------------------
        # Flat list of unique children with their owning scramble
        child_states_flat: list[tuple] = []
        child_move_flat: list[int] = []
        child_owner_flat: list[int] = []

        for i in active:
            seen: dict = {}  # state_key -> index in scramble_children[i] deduped
            deduped: list[tuple[tuple, int]] = []
            for child_state, move_idx in scramble_children[i]:
                k = _state_key(child_state)
                if k not in seen:
                    seen[k] = len(deduped)
                    deduped.append((child_state, move_idx))
            scramble_children[i] = deduped
            for child_state, move_idx in deduped:
                child_states_flat.append(child_state)
                child_move_flat.append(move_idx)
                child_owner_flat.append(i)

        if not child_states_flat:
            break

        # Score children with the value head
        c_batch = len(child_states_flat)
        c_curr_cp, c_curr_ct, c_curr_ep, c_curr_ef = states_to_arrays(child_states_flat)
        c_goal_cp, c_goal_ct, c_goal_ep, c_goal_ef = states_to_arrays(
            [goal_state] * c_batch
        )
        c_t_val = _t_value(t_mode, scramble_depth, step + 1, t_const)
        c_t = mx.array([c_t_val] * c_batch, dtype=mx.int32) if c_t_val is not None else None

        _, child_values = model(
            goal=(c_goal_cp, c_goal_ct, c_goal_ep, c_goal_ef),
            curr=(c_curr_cp, c_curr_ct, c_curr_ep, c_curr_ef),
            t=c_t,
            return_value=True,
        )
        mx.eval(child_values)
        child_values_list = child_values.tolist()

        # ------------------------------------------------------------------
        # Step 3: for each scramble, rank children by value (lowest = closer
        # to solved) and keep top beam_width.
        # ------------------------------------------------------------------
        # Collect (value, state, move_idx) per scramble
        scramble_ranked: dict[int, list[tuple[float, tuple, int]]] = {i: [] for i in active}
        for j, i in enumerate(child_owner_flat):
            v = child_values_list[j]
            state = child_states_flat[j]
            move_idx = child_move_flat[j]
            scramble_ranked[i].append((v, state, move_idx))

        new_beams: dict[int, list[tuple[tuple, int]]] = {}
        for i in active:
            ranked = scramble_ranked[i]
            ranked.sort(key=lambda x: x[0])  # ascending: lower value = better
            ranked = ranked[:beam_width]
            new_beams[i] = [(st, mi) for _, st, mi in ranked]

        # Check for solved states; commit new beams
        for i in active:
            for state, _mi in new_beams[i]:
                if cube_solved(state):
                    solved_step[i] = step + 1
                    break
            beams[i] = new_beams[i]

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
    beam_width: int = 0,
    t_mode: str = "countdown",
    t_const: int | None = None,
    search: str = "auto",
    **model_cfg,
) -> dict:
    """Evaluate a checkpoint on n scrambles.

    Parameters
    ----------
    ckpt_path     : path to the .npz checkpoint
    n             : number of scrambles to evaluate
    scramble_depth: depth of each scramble
    max_steps     : maximum rollout steps (0/None = auto: max(60, scramble_depth*6))
    seed          : random seed for scramble generation
    beam_width    : if > 0, use beam search with this width; 0 = greedy
    t_mode        : 'countdown' | 'const' | 'none'
    t_const       : fixed t value for t_mode='const' (default = scramble_depth)
    search        : 'auto' | 'greedy' | 'beam' | 'value-beam'
                    'auto' = beam if beam_width>0 else greedy (legacy behaviour)
    **model_cfg   : forwarded to load_model_auto (d_model, n_layers, etc.)

    Returns
    -------
    dict with keys: success_rate, n_solved, n, avg_steps_solved,
                    median_steps_solved, scramble_depth, max_steps, beam_width,
                    search
    """
    if max_steps is None or max_steps <= 0:
        max_steps = max(60, scramble_depth * 6)

    if t_const is None:
        t_const = scramble_depth

    model = load_model_auto(ckpt_path, **model_cfg)
    scrambles = _generate_scrambles(n, scramble_depth, seed=seed)

    # Resolve effective search mode
    if search == "auto":
        effective_search = "beam" if beam_width > 0 else "greedy"
    else:
        effective_search = search

    if effective_search == "value-beam":
        if beam_width <= 0:
            beam_width = 8  # sensible default if user forgot --beam
        solved_mask, steps = rollout_value_beam(
            model, scrambles, scramble_depth, max_steps, beam_width,
            t_mode=t_mode, t_const=t_const,
        )
    elif effective_search == "beam":
        solved_mask, steps = rollout_beam(
            model, scrambles, scramble_depth, max_steps, beam_width,
            t_mode=t_mode, t_const=t_const,
        )
    else:
        solved_mask, steps = rollout(
            model, scrambles, scramble_depth, max_steps,
            t_mode=t_mode, t_const=t_const,
        )

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
        "beam_width": beam_width,
        "search": effective_search,
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
        help="Max rollout steps (0 = auto: max(60, scramble_depth * 6)).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run the CFOP baseline on the same scrambles.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=0,
        metavar="WIDTH",
        help="Beam width for beam-search rollout (0 = greedy, default).",
    )
    parser.add_argument(
        "--search",
        choices=["greedy", "beam", "value-beam"],
        default=None,
        help=(
            "Search strategy: 'greedy' (argmax), 'beam' (cumulative-logprob beam), "
            "'value-beam' (policy-proposal + value-ranking). "
            "Default: 'beam' if --beam>0 else 'greedy' (legacy behaviour)."
        ),
    )
    # t-conditioning mode
    parser.add_argument(
        "--t-mode",
        choices=["countdown", "const", "none"],
        default="countdown",
        help=(
            "How to condition the model on noise level t at each step. "
            "'countdown': t = max(1, scramble_depth - step) [default]. "
            "'const': t = --t-const every step. "
            "'none': pass t=None to the model (unconditional)."
        ),
    )
    parser.add_argument(
        "--t-const",
        type=int,
        default=None,
        help="Fixed t value used when --t-mode const (default = scramble_depth).",
    )
    # Model architecture args (for non-default checkpoints)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--t-max", type=int, default=100)

    args = parser.parse_args()

    max_steps = args.max_steps if args.max_steps > 0 else None
    effective_max = max_steps if max_steps is not None else max(60, args.scramble_depth * 6)
    t_const = args.t_const if args.t_const is not None else args.scramble_depth

    model_cfg = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        t_max=args.t_max,
    )

    # Resolve search mode for display
    search_arg = args.search if args.search is not None else "auto"
    if search_arg == "auto":
        effective_search_display = "beam" if args.beam > 0 else "greedy"
    else:
        effective_search_display = search_arg
    beam_mode = args.beam > 0
    print(
        f"Evaluating: n={args.n}, scramble_depth={args.scramble_depth}, "
        f"max_steps={effective_max}, seed={args.seed}, t_mode={args.t_mode}"
        + (f", t_const={t_const}" if args.t_mode == "const" else "")
        + (f", beam_width={args.beam}" if beam_mode else "")
        + f", search={effective_search_display}"
    )

    rows = []

    if args.ckpts:
        for ckpt_path in args.ckpts:
            search_label = effective_search_display
            if beam_mode:
                label = f"{ckpt_path} ({search_label},beam={args.beam})"
            else:
                label = f"{ckpt_path} ({search_label})"
            print(f"  loading {ckpt_path} ...", flush=True)
            result = evaluate(
                ckpt_path,
                n=args.n,
                scramble_depth=args.scramble_depth,
                max_steps=max_steps,
                seed=args.seed,
                beam_width=args.beam,
                t_mode=args.t_mode,
                t_const=t_const,
                search=search_arg,
                **model_cfg,
            )
            rows.append(
                {
                    "label": label,
                    "success_rate": result["success_rate"],
                    "avg_steps": result["avg_steps_solved"],
                    "median_steps": result["median_steps_solved"],
                }
            )
            print(
                f"    solved {result['n_solved']}/{result['n']} "
                f"({result['success_rate']:.1%})"
                f"  [search={result['search']}]",
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
