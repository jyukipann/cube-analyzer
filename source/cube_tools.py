"""Cube tool kernel for an LLM agent (M0).

Wraps the verified MLX cube representation (state.py / vis_util.py) and the
learned cost-to-go value head (model/solver.py via infer.py) behind a small,
LLM-friendly tool surface:

    session.scramble(depth)      -> observation
    session.reset()              -> observation
    session.apply("R U R' U'")   -> commit moves, observation
    session.simulate("R U R'")   -> effect on a SCRATCH copy (no commit)
    session.observe()            -> ASCII net + solved-piece counts
    session.inverse("R U2 F'")   -> "F U2 R'"
    session.distance()           -> learned cost-to-go (M3, lazy-loaded)

State is held as ground truth in the kernel; the LLM never has to track it.

Move notation
-------------
Faces: U D L R F B.  Suffix: '' = quarter, '2' = half, "'" = inverse quarter.
18 moves total.  Index convention matches data.py: face*3 + (turns-1) with face
order U=0, D=1, L=2, R=3, F=4, B=5.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import mlx.core as mx

_SRC = Path(__file__).resolve().parent
for _d in (str(_SRC), str(_SRC / "cube")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from state import State, MOVES                 # noqa: E402
from vis_util import state_to_net, print_net   # noqa: E402

# ---------------------------------------------------------------------------
# Move notation <-> precomputed State
# ---------------------------------------------------------------------------

_FACE_ORDER = ['U', 'D', 'L', 'R', 'F', 'B']
_SUFFIX = {1: '', 2: '2', 3: "'"}

# NOTATION[idx] -> "R", "R2", "R'" ; _PARSE["R'"] -> idx
NOTATION: list[str] = []
_MOVE_STATES: list[State] = []
_PARSE: dict[str, int] = {}

for _fi, _face in enumerate(_FACE_ORDER):
    _q = MOVES[_face]
    for _turns in (1, 2, 3):
        _s = _q
        for _ in range(_turns - 1):
            _s = _s @ _q
        _idx = _fi * 3 + (_turns - 1)
        _tok = _face + _SUFFIX[_turns]
        NOTATION.append(_tok)
        _MOVE_STATES.append(_s)
        _PARSE[_tok] = _idx

# Inverse-token table: inverse of (face, turns) is (face, 4 - turns)
_INV_TOKEN: dict[str, str] = {}
for _fi, _face in enumerate(_FACE_ORDER):
    for _turns in (1, 2, 3):
        _tok = _face + _SUFFIX[_turns]
        _INV_TOKEN[_tok] = _face + _SUFFIX[4 - _turns]


class MoveParseError(ValueError):
    """Raised when a move string cannot be parsed."""


def parse_moves(moves: str) -> list[int]:
    """Parse a whitespace-separated move string into a list of move indices.

    Accepts tokens like 'R', "R'", 'R2'.  Lowercase is upcased.  Raises
    MoveParseError on any unknown token (with the offending token named).
    """
    if not moves or not moves.strip():
        return []
    out: list[int] = []
    for raw in moves.replace(",", " ").split():
        tok = raw.strip().upper().replace("’", "'")  # normalize unicode prime
        if tok not in _PARSE:
            raise MoveParseError(
                f"unknown move '{raw}'. Valid moves: {', '.join(NOTATION)}"
            )
        out.append(_PARSE[tok])
    return out


def invert_moves(moves: str) -> str:
    """Return the move string that exactly undoes `moves` (reversed + inverted)."""
    idxs = parse_moves(moves)
    toks = [NOTATION[i] for i in idxs]
    return ' '.join(_INV_TOKEN[t] for t in reversed(toks))


def _apply_indices(state: State, idxs: list[int]) -> State:
    s = state
    for i in idxs:
        s = s @ _MOVE_STATES[i]
    return s


# ---------------------------------------------------------------------------
# Solved-piece metrics
# ---------------------------------------------------------------------------

def _solved_counts(state: State) -> dict:
    cp = [int(x) for x in state.corner_positions.tolist()]
    ct = [int(x) for x in state.twist_co.tolist()]
    ep = [int(x) for x in state.edge_positions.tolist()]
    ef = [int(x) for x in state.twist_eo.tolist()]
    corners_ok = sum(1 for i in range(8) if cp[i] == i and ct[i] == 0)
    edges_ok = sum(1 for i in range(12) if ep[i] == i and ef[i] == 0)
    return {
        "corners_solved": corners_ok,   # 0..8 (placed AND oriented)
        "edges_solved": edges_ok,        # 0..12
        "pieces_solved": corners_ok + edges_ok,  # 0..20
        "is_solved": corners_ok == 8 and edges_ok == 12,
    }


def render_net(state: State) -> str:
    """ASCII unfolded-net rendering of a state (the LLM's eyes)."""
    return print_net(state_to_net(state))


# ---------------------------------------------------------------------------
# Cube session (ground-truth state held here)
# ---------------------------------------------------------------------------

class CubeSession:
    """A single interactive cube the LLM agent manipulates via tools."""

    def __init__(self, seed: int | None = None):
        self.state = State()                 # solved
        self.rng = random.Random(seed)
        self.scramble_moves: list[int] = []   # the scramble that was applied
        self.history: list[int] = []          # committed solving moves
        self._value_model = None              # lazy (M3)

    # -- setup ---------------------------------------------------------------
    def reset(self) -> dict:
        """Return the cube to solved and clear history."""
        self.state = State()
        self.scramble_moves = []
        self.history = []
        return self.observe()

    def scramble(self, depth: int = 5) -> dict:
        """Reset, then apply `depth` random moves (avoiding trivial cancels)."""
        self.state = State()
        self.history = []
        idxs: list[int] = []
        prev_face = -1
        for _ in range(depth):
            i = self.rng.randrange(18)
            # avoid same-face-back-to-back (keeps the scramble depth honest)
            while (i // 3) == prev_face:
                i = self.rng.randrange(18)
            idxs.append(i)
            prev_face = i // 3
        self.scramble_moves = idxs
        self.state = _apply_indices(self.state, idxs)
        obs = self.observe()
        obs["scramble"] = ' '.join(NOTATION[i] for i in idxs)
        obs["scramble_depth"] = depth
        return obs

    # -- actions -------------------------------------------------------------
    def apply(self, moves: str) -> dict:
        """Commit `moves` to the cube and return the new observation."""
        idxs = parse_moves(moves)
        self.state = _apply_indices(self.state, idxs)
        self.history.extend(idxs)
        obs = self.observe()
        obs["applied"] = ' '.join(NOTATION[i] for i in idxs)
        return obs

    def simulate(self, moves: str) -> dict:
        """Show the effect of `moves` on a SCRATCH copy without committing.

        Returns before/after solved-piece counts and the resulting net, so the
        agent can experiment with a candidate macro safely.
        """
        idxs = parse_moves(moves)
        scratch = _apply_indices(self.state, idxs)
        before = _solved_counts(self.state)
        after = _solved_counts(scratch)
        return {
            "moves": ' '.join(NOTATION[i] for i in idxs),
            "net_after": render_net(scratch),
            "pieces_solved_before": before["pieces_solved"],
            "pieces_solved_after": after["pieces_solved"],
            "delta_pieces_solved": after["pieces_solved"] - before["pieces_solved"],
            "would_solve": after["is_solved"],
            "committed": False,
        }

    # -- observation ---------------------------------------------------------
    def observe(self) -> dict:
        c = _solved_counts(self.state)
        return {
            "net": render_net(self.state),
            "corners_solved": c["corners_solved"],
            "edges_solved": c["edges_solved"],
            "pieces_solved": c["pieces_solved"],
            "is_solved": c["is_solved"],
            "moves_made": len(self.history),
        }

    def inverse(self, moves: str) -> dict:
        return {"moves": moves, "inverse": invert_moves(moves)}

    # -- learned intuition (M3) ---------------------------------------------
    def distance(self) -> dict:
        """Learned cost-to-go estimate for the current state (lower = closer)."""
        val = self._value(self.state)
        return {"estimated_moves_to_solve": round(val, 2)}

    def _value(self, state: State) -> float:
        if self._value_model is None:
            from infer import load_model_auto  # lazy import (loads MLX model)
            self._value_model = load_model_auto(
                str(_SRC.parent / "runs" / "hindsight_not" / "latest.npz")
            )
        m = self._value_model
        cp = mx.array([[int(x) for x in state.corner_positions.tolist()]], dtype=mx.int32)
        ct = mx.array([[int(x) for x in state.twist_co.tolist()]], dtype=mx.int32)
        ep = mx.array([[int(x) for x in state.edge_positions.tolist()]], dtype=mx.int32)
        ef = mx.array([[int(x) for x in state.twist_eo.tolist()]], dtype=mx.int32)
        gcp = mx.array([list(range(8))], dtype=mx.int32)
        gct = mx.zeros((1, 8), dtype=mx.int32)
        gep = mx.array([list(range(12))], dtype=mx.int32)
        gef = mx.zeros((1, 12), dtype=mx.int32)
        _, value = m(goal=(gcp, gct, gep, gef), curr=(cp, ct, ep, ef),
                     t=None, return_value=True)
        mx.eval(value)
        return float(value.tolist()[0])


# ---------------------------------------------------------------------------
# Manual smoke check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    s = CubeSession(seed=0)
    print("solved net:\n" + s.observe()["net"])
    print("\nNOTATION:", NOTATION)
    # sanity: a move and its inverse return to solved
    s.apply("R U R' U'")
    print("\nafter R U R' U'  pieces_solved =", s.observe()["pieces_solved"])
    s.apply(invert_moves("R U R' U'"))
    print("undone -> is_solved =", s.observe()["is_solved"])
    # scramble then undo with the exact inverse
    obs = s.scramble(8)
    print("\nscramble:", obs["scramble"], " pieces_solved =", obs["pieces_solved"])
    undo = invert_moves(' '.join(NOTATION[i] for i in s.scramble_moves))
    s.apply(undo)
    print("inverse scramble -> is_solved =", s.observe()["is_solved"])
    # simulate does not commit
    s.scramble(5)
    before = s.observe()["pieces_solved"]
    sim = s.simulate("R U R' U'")
    print("\nsimulate delta =", sim["delta_pieces_solved"],
          " committed? state unchanged:", s.observe()["pieces_solved"] == before)
