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
