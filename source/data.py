"""Training batch generation for the cube diffusion solver.

Inner loops are pure Python (no MLX) to avoid lazy-evaluation graph
accumulation during sequential state transitions. The 18 precomputed move
tables (_MOVES_PY) are derived from the MLX MOVES once at import time.

Move index convention
---------------------
Faces ordered: U=0, D=1, L=2, R=3, F=4, B=5
Turns:         1-turn (+0), 2-turn (+1), 3-turn (+2) within each face block
Index:         face_idx * 3 + (turns - 1)    range 0..17

Inverse:       (face, turns) -> (face, 4 - turns)
               i.e. 1 <-> 3, 2 <-> 2
"""

import random
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent / "cube"))
from state import MOVES  # noqa: E402

# ---------------------------------------------------------------------------
# Pure-Python state representation
# (cp[8], ct[8], ep[12], ef[12]) all indexed by slot
# ct = twist 0/1/2;  ef = flip 0/1

_IDENTITY: tuple = (list(range(8)), [0] * 8, list(range(12)), [0] * 12)


def _to_py(mlx_state) -> tuple:
    """Convert MLX State to pure-Python tuple (forced evaluation)."""
    return (
        [int(x) for x in mlx_state.corner_positions.tolist()],
        [int(x) for x in mlx_state.twist_co.tolist()],
        [int(x) for x in mlx_state.edge_positions.tolist()],
        [int(x) for x in mlx_state.twist_eo.tolist()],
    )


def _compose(a: tuple, b: tuple) -> tuple:
    """Wreath-product composition a @ b in pure Python."""
    acp, act, aep, aef = a
    bcp, bct, bep, bef = b
    return (
        [acp[bcp[i]] for i in range(8)],
        [(act[bcp[i]] + bct[i]) % 3 for i in range(8)],
        [aep[bep[i]] for i in range(12)],
        [(aef[bep[i]] + bef[i]) % 2 for i in range(12)],
    )


# Precompute all 18 moves as pure-Python tuples
_FACE_ORDER = ['U', 'D', 'L', 'R', 'F', 'B']
_MOVES_PY: list[tuple] = []
for _face in _FACE_ORDER:
    _m1 = _to_py(MOVES[_face])
    _m2 = _compose(_m1, _m1)
    _m3 = _compose(_m2, _m1)
    _MOVES_PY.extend([_m1, _m2, _m3])

# Inverse index table: _INV_IDX[i] = index of move that undoes move i
_INV_IDX: list[int] = []
for _i in range(18):
    _fi = (_i // 3) * 3
    _t = _i % 3 + 1       # turns: 1, 2, 3
    _inv_t = 4 - _t        # inverse turns: 3, 2, 1
    _INV_IDX.append(_fi + (_inv_t - 1))


# ---------------------------------------------------------------------------

def generate_batch(
    batch_size: int,
    t_max: int = 100,
    goal_py: tuple = _IDENTITY,
) -> dict[str, mx.array]:
    """Generate one training batch as a dict of int32 MLX arrays.

    Each sample:
      - start from goal_py
      - apply t ~ Uniform(1, t_max) random moves from the 18-move vocab
      - target = index of the inverse of the last applied move

    Keys returned
    -------------
    gcp, gct, gep, gef  goal state   (shapes [B,8], [B,8], [B,12], [B,12])
    ccp, cct, cep, cef  noisy state  (same shapes)
    t                   noise level  [B]
    target              move index to undo last step  [B]
    """
    gcp, gct, gep, gef = goal_py

    rows: dict[str, list] = {k: [] for k in (
        'gcp', 'gct', 'gep', 'gef',
        'ccp', 'cct', 'cep', 'cef',
        't', 'target',
    )}

    for _ in range(batch_size):
        t_val = random.randint(1, t_max)
        state = goal_py
        last_idx = 0
        for _ in range(t_val):
            last_idx = random.randrange(18)
            state = _compose(state, _MOVES_PY[last_idx])

        rows['gcp'].append(gcp);        rows['gct'].append(gct)
        rows['gep'].append(gep);        rows['gef'].append(gef)
        rows['ccp'].append(state[0]);   rows['cct'].append(state[1])
        rows['cep'].append(state[2]);   rows['cef'].append(state[3])
        rows['t'].append(t_val)
        rows['target'].append(_INV_IDX[last_idx])

    return {k: mx.array(v, dtype=mx.int32) for k, v in rows.items()}


def cfop_batch(
    batch_size: int,
    scramble_depth: int = 25,
    t_max: int = 100,
) -> dict[str, mx.array]:
    """Generate a behavioral-cloning batch from the CFOP solver.

    Each sample is one step along a CFOP solution trajectory:
      - current_state : the cube state at that step
      - target        : the move index the CFOP solver takes from that state
      - t             : moves remaining in the solution (distance-to-go),
                        clamped to [1, t_max]
      - goal          : _IDENTITY (solved cube) for every sample

    Samples are collected across multiple scrambles until batch_size is reached.

    Keys returned
    -------------
    gcp, gct, gep, gef  goal state   (shapes [B,8], [B,8], [B,12], [B,12])
    ccp, cct, cep, cef  current state (same shapes)
    t                   distance-to-go  [B]
    target              move index taken by CFOP solver  [B]
    """
    import sys as _sys
    from pathlib import Path as _Path
    _src_dir = str(_Path(__file__).parent)
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)
    import cfop as _cfop

    goal_py = _IDENTITY
    gcp, gct, gep, gef = goal_py

    rows: dict[str, list] = {k: [] for k in (
        'gcp', 'gct', 'gep', 'gef',
        'ccp', 'cct', 'cep', 'cef',
        't', 'target',
    )}

    collected = 0
    while collected < batch_size:
        # Scramble from identity
        state = _IDENTITY
        for _ in range(scramble_depth):
            state = _compose(state, _MOVES_PY[random.randrange(18)])

        # Ask CFOP solver for the full solution move list
        try:
            solution = _cfop.solve(state)
        except RuntimeError:
            # Skip unsolvable states (shouldn't happen, but be defensive)
            continue

        if not solution:
            # Already solved — no training signal
            continue

        # Walk along the solution trajectory
        current = state
        n_remaining = len(solution)
        for step_idx, move_idx in enumerate(solution):
            if collected >= batch_size:
                break
            t_val = min(max(n_remaining - step_idx, 1), t_max)
            rows['gcp'].append(gcp);          rows['gct'].append(gct)
            rows['gep'].append(gep);          rows['gef'].append(gef)
            rows['ccp'].append(current[0]);   rows['cct'].append(current[1])
            rows['cep'].append(current[2]);   rows['cef'].append(current[3])
            rows['t'].append(t_val)
            rows['target'].append(move_idx)
            collected += 1
            current = _compose(current, _MOVES_PY[move_idx])

    return {k: mx.array(v, dtype=mx.int32) for k, v in rows.items()}


def build_cfop_pool(
    n_samples: int,
    scramble_depth: int = 25,
    t_max: int = 100,
    cache_path: str | None = None,
    verbose: bool = True,
    min_depth: int | None = None,
    randomize: bool = False,
) -> dict[str, mx.array]:
    """Build a large pool of behavioral-cloning samples from the CFOP solver.

    Repeatedly scrambles the identity cube, solves it with CFOP, and walks
    the solution trajectory to accumulate (current_state, next_move) samples.
    goal is always _IDENTITY; t = moves-remaining clamped to [1, t_max].

    Parameters
    ----------
    n_samples     : total samples to accumulate
    scramble_depth: number of random moves used to scramble each cube (max depth
                    when min_depth is set)
    t_max         : maximum t value (distance-to-go is clamped to [1, t_max])
    cache_path    : if given, save the finished pool as an .npz file here
    verbose       : print progress every ~10000 samples
    min_depth     : if set, each scramble uses a random depth in
                    [min_depth, scramble_depth]; None = fixed scramble_depth
    randomize     : if True, pass randomize=True to cfop.solve() so that
                    pair order and AUF choices vary per scramble

    Returns
    -------
    dict with keys gcp,gct,gep,gef,ccp,cct,cep,cef,t,target;
    each value is mx.int32 array of shape [n_samples, ...].
    """
    _src_dir = str(Path(__file__).parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import cfop as _cfop

    goal_py = _IDENTITY
    gcp, gct, gep, gef = goal_py

    rows: dict[str, list] = {k: [] for k in (
        'gcp', 'gct', 'gep', 'gef',
        'ccp', 'cct', 'cep', 'cef',
        't', 'target',
    )}

    collected = 0
    n_scrambles = 0
    t0 = time.time()
    last_report = 0
    # Single seeded RNG for reproducibility; used both for scramble depth
    # selection and (if randomize=True) passed to cfop.solve per scramble.
    _rng = random.Random(42)

    while collected < n_samples:
        # Determine scramble depth for this cube
        depth = (_rng.randint(min_depth, scramble_depth)
                 if min_depth is not None
                 else scramble_depth)

        # Scramble from identity
        state = _IDENTITY
        for _ in range(depth):
            state = _compose(state, _MOVES_PY[_rng.randrange(18)])

        # Per-scramble solver RNG (only used when randomize=True)
        solve_rng = random.Random(_rng.randrange(2**32)) if randomize else None

        try:
            solution = _cfop.solve(state, randomize=randomize, rng=solve_rng)
        except RuntimeError:
            continue

        if not solution:
            continue

        n_scrambles += 1
        current = state
        n_remaining = len(solution)
        for step_idx, move_idx in enumerate(solution):
            if collected >= n_samples:
                break
            t_val = min(max(n_remaining - step_idx, 1), t_max)
            rows['gcp'].append(gcp);          rows['gct'].append(gct)
            rows['gep'].append(gep);          rows['gef'].append(gef)
            rows['ccp'].append(current[0]);   rows['cct'].append(current[1])
            rows['cep'].append(current[2]);   rows['cef'].append(current[3])
            rows['t'].append(t_val)
            rows['target'].append(move_idx)
            collected += 1
            current = _compose(current, _MOVES_PY[move_idx])

        if verbose and collected - last_report >= 10000:
            elapsed = time.time() - t0
            print(
                f"pool: {collected}/{n_samples} samples "
                f"({elapsed:.1f}s, {n_scrambles} scrambles)",
                flush=True,
            )
            last_report = collected

    pool = {k: mx.array(v, dtype=mx.int32) for k, v in rows.items()}

    if cache_path is not None:
        mx.savez(cache_path, **pool)
        if verbose:
            print(f"pool: saved {n_samples} samples to {cache_path}", flush=True)

    return pool


def load_cfop_pool(
    n_samples: int,
    scramble_depth: int = 25,
    t_max: int = 100,
    cache_path: str = "source/.cfop_pool.npz",
    verbose: bool = True,
    min_depth: int | None = None,
    randomize: bool = False,
) -> dict[str, mx.array]:
    """Load a CFOP sample pool from cache, or build (and save) it if needed.

    If the .npz at cache_path exists and contains at least n_samples rows,
    it is loaded and sliced to exactly n_samples. Otherwise the pool is built
    from scratch and saved to cache_path for future reuse.

    Parameters
    ----------
    n_samples  : number of samples required
    cache_path : path to the .npz cache file (used as the base path; a diverse
                 pool automatically gets a distinct suffix so it does not
                 collide with the plain pool cache)
    min_depth  : if set, scramble depth varies randomly in [min_depth, scramble_depth]
    randomize  : if True, solver introduces pair-order and AUF diversity;
                 a distinct cache file is used (never collides with plain pool)
    (other params forwarded to build_cfop_pool when a rebuild is needed)
    """
    # Derive a cache path that encodes diversity settings so diverse and plain
    # pools never share the same file.  When randomize=True we skip caching
    # entirely (randomized pools intentionally vary per run).
    effective_cache: str | None
    if randomize:
        effective_cache = None  # do not cache randomized pools
    elif min_depth is not None:
        # Embed min_depth into the filename stem to avoid collisions
        p_obj = Path(cache_path)
        stem = p_obj.stem + f"_min{min_depth}_max{scramble_depth}"
        effective_cache = str(p_obj.with_name(stem + p_obj.suffix))
    else:
        effective_cache = cache_path

    if effective_cache is not None:
        p = Path(effective_cache)
        if p.exists():
            try:
                loaded = mx.load(str(p))
                # Check row count using one of the arrays
                sample_arr = loaded.get('t')
                if sample_arr is not None and sample_arr.shape[0] >= n_samples:
                    if verbose:
                        print(
                            f"pool: loaded {sample_arr.shape[0]} samples from "
                            f"{effective_cache}, slicing to {n_samples}",
                            flush=True,
                        )
                    return {k: v[:n_samples] for k, v in loaded.items()}
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"pool: cache load failed ({exc}); rebuilding",
                          flush=True)

    return build_cfop_pool(
        n_samples,
        scramble_depth=scramble_depth,
        t_max=t_max,
        cache_path=effective_cache,
        verbose=verbose,
        min_depth=min_depth,
        randomize=randomize,
    )
