"""CFOP solver for behavioral cloning data generation.

Stages:
  1. Cross — solve 4 D-layer edges        (BFS, all 18 moves, depth ≤ 8)
  2. F2L   — solve 4 corner-edge pairs     (BFS per pair, no-D moves)
  3. OLL   — orient last layer             (precomputed LL table)
  4. PLL   — permute last layer            (precomputed LL table)

Why the LL tables are built over a *subgroup*
---------------------------------------------
After Cross+F2L the first two layers are fixed; only the U layer (corners
0-3, edges 4-7) varies. That last-layer (LL) group has 62 208 elements.
We build solution tables by BFS over *that* group, using standard
F2L-preserving algorithms (Sune, T-perm, ...) as macro edges. A naive BFS
over all 18 single moves keyed by U-layer signature instead wanders through
~2·10^7 whole-cube states and never terminates in pure Python — that was the
bug this design replaces.

Everything is observable: table construction prints node counts + timing,
and results are cached to disk so reruns load instantly.

State representation: pure-Python (cp[8], ct[8], ep[12], ef[12]) tuples,
identical to data.py. ct = twist 0/1/2; ef = flip 0/1.

Move index convention (same as data.py):
  Faces: U=0, D=1, L=2, R=3, F=4, B=5
  Index: face_idx * 3 + (turns - 1),  range 0..17
"""

import pickle
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "cube"))
from state import MOVES  # noqa: E402

# ---------------------------------------------------------------------------
# Pure-Python primitives (mirrors data.py)
# ---------------------------------------------------------------------------

_IDENTITY: tuple = (list(range(8)), [0] * 8, list(range(12)), [0] * 12)


def _to_py(mlx_state) -> tuple:
    return (
        [int(x) for x in mlx_state.corner_positions.tolist()],
        [int(x) for x in mlx_state.twist_co.tolist()],
        [int(x) for x in mlx_state.edge_positions.tolist()],
        [int(x) for x in mlx_state.twist_eo.tolist()],
    )


def _compose(a: tuple, b: tuple) -> tuple:
    acp, act, aep, aef = a
    bcp, bct, bep, bef = b
    return (
        [acp[bcp[i]] for i in range(8)],
        [(act[bcp[i]] + bct[i]) % 3 for i in range(8)],
        [aep[bep[i]] for i in range(12)],
        [(aef[bep[i]] + bef[i]) % 2 for i in range(12)],
    )


_FACE_ORDER = ['U', 'D', 'L', 'R', 'F', 'B']
_MOVES_PY: list[tuple] = []
for _face in _FACE_ORDER:
    _m1 = _to_py(MOVES[_face])
    _m2 = _compose(_m1, _m1)
    _m3 = _compose(_m2, _m1)
    _MOVES_PY.extend([_m1, _m2, _m3])

# Inverse index table: _INV_IDX[i] undoes move i
_INV_IDX: list[int] = []
for _i in range(18):
    _fi = (_i // 3) * 3
    _t = _i % 3 + 1
    _INV_IDX.append(_fi + (4 - _t - 1))


def _invert_seq(seq: list[int]) -> list[int]:
    """Inverse of a move-index sequence."""
    return [_INV_IDX[m] for m in reversed(seq)]


def _apply_moves(state: tuple, moves: list[int]) -> tuple:
    for mi in moves:
        state = _compose(state, _MOVES_PY[mi])
    return state


def _state_key(s: tuple) -> tuple:
    cp, ct, ep, ef = s
    return (tuple(cp), tuple(ct), tuple(ep), tuple(ef))


# ---------------------------------------------------------------------------
# Slot geometry
# ---------------------------------------------------------------------------
# Corners: C0=LUB C1=RBU C2=RUF C3=LFU (U) | C4=LBD C5=RDB C6=RFD C7=LDF (D)
# Edges:   E0=LB E1=RB E2=RF E3=LF (mid) | E4=UB E5=RU E6=UF E7=LU (U)
#          E8=DB E9=RD E10=DF E11=LD (D)

_CROSS_EDGES = [8, 9, 10, 11]              # D-layer edge slots
_F2L_PAIRS = [(6, 2), (7, 3), (5, 1), (4, 0)]  # (corner_slot, edge_slot): FR FL BR BL

# Per-pair move set = the slot's two adjacent side faces + U.
# Keeps branching at 9 and the pair pieces reachable. Face move-index blocks:
#   U:0-2  L:6-8  R:9-11  F:12-14  B:15-17
_U = [0, 1, 2]
_SLOT_MOVES = {
    0: _U + [9, 10, 11] + [12, 13, 14],   # FR: U R F
    1: _U + [6, 7, 8] + [12, 13, 14],     # FL: U L F
    2: _U + [9, 10, 11] + [15, 16, 17],   # BR: U R B
    3: _U + [6, 7, 8] + [15, 16, 17],     # BL: U L B
}
# Fallback move set (everything except D, which breaks the cross)
_F2L_MOVES = [i for i in range(18) if i // 3 != 1]


# ---------------------------------------------------------------------------
# Stage goal predicates
# ---------------------------------------------------------------------------

def cross_solved(s: tuple) -> bool:
    _, _, ep, ef = s
    return all(ep[slot] == slot and ef[slot] == 0 for slot in _CROSS_EDGES)


def _f2l_pair_solved(s: tuple, pair_idx: int) -> bool:
    cp, ct, ep, ef = s
    c, e = _F2L_PAIRS[pair_idx]
    return cp[c] == c and ct[c] == 0 and ep[e] == e and ef[e] == 0


def f2l_solved(s: tuple) -> bool:
    return all(_f2l_pair_solved(s, i) for i in range(4))


def oll_solved(s: tuple) -> bool:
    """U-layer corners (0-3) twist 0 and U-layer edges (4-7) flip 0."""
    _, ct, _, ef = s
    return all(ct[i] == 0 for i in range(4)) and all(ef[i] == 0 for i in range(4, 8))


def cube_solved(s: tuple) -> bool:
    return (s[0] == list(range(8)) and s[1] == [0] * 8
            and s[2] == list(range(12)) and s[3] == [0] * 12)


# ---------------------------------------------------------------------------
# Last-layer (LL) signature: only U-layer corners 0-3 and edges 4-7 vary
# ---------------------------------------------------------------------------

def _ll_key(s: tuple) -> tuple:
    cp, ct, ep, ef = s
    return (cp[0], cp[1], cp[2], cp[3], ct[0], ct[1], ct[2], ct[3],
            ep[4], ep[5], ep[6], ep[7], ef[4], ef[5], ef[6], ef[7])


def _cross_key(s: tuple) -> tuple:
    """Signature of the 4 D-layer cross edges: (slot, flip) of pieces 8-11.
    Depends only on the cross edges, so a table keyed by it solves the cross
    for any whole-cube state sharing the signature."""
    _, _, ep, ef = s
    out = []
    for p in (8, 9, 10, 11):
        slot = ep.index(p)
        out.append((slot, ef[slot]))
    return tuple(out)


def _ll_state_from_key(k: tuple) -> tuple:
    cp = [k[0], k[1], k[2], k[3], 4, 5, 6, 7]
    ct = [k[4], k[5], k[6], k[7], 0, 0, 0, 0]
    ep = [0, 1, 2, 3, k[8], k[9], k[10], k[11], 8, 9, 10, 11]
    ef = [0, 0, 0, 0, k[12], k[13], k[14], k[15], 0, 0, 0, 0]
    return (cp, ct, ep, ef)


# ---------------------------------------------------------------------------
# LL macro generators: standard CFOP algorithms (cross on bottom).
# Each MUST preserve the first two layers — asserted at build time.
# ---------------------------------------------------------------------------

def _seq(notation: str) -> list[int]:
    """'R U R\\'' -> [move indices]."""
    face_idx = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'F': 4, 'B': 5}
    out = []
    for tok in notation.split():
        f = face_idx[tok[0]]
        turns = 1
        if tok.endswith("2"):
            turns = 2
        elif tok.endswith("'"):
            turns = 3
        out.append(f * 3 + (turns - 1))
    return out


_GENERATORS_NOTATION = {
    "U":         "U",
    "Sune":      "R U R' U R U2 R'",
    "Antisune":  "R U2 R' U' R U' R'",
    "T-perm":    "R U R' U' R' F R2 U' R' U' R U R' F'",
    "Y-perm":    "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "Ua-perm":   "R U' R U R U R U' R' U' R2",
    "F-EO":      "F R U R' U' F'",
    "H-perm":    "R2 U2 R U2 R2 U2 R2 U2 R U2 R2",
}


def _build_generators() -> list[tuple[tuple, list[int]]]:
    """Return [(effect_tuple, move_seq), ...] for each generator and its inverse.
    Asserts every generator preserves cross + F2L."""
    gens: list[tuple[tuple, list[int]]] = []
    for name, notation in _GENERATORS_NOTATION.items():
        seq = _seq(notation)
        eff = _apply_moves(_IDENTITY, seq)
        if not (cross_solved(eff) and f2l_solved(eff)):
            raise AssertionError(f"generator {name!r} does not preserve F2L")
        gens.append((eff, seq))
        inv = _invert_seq(seq)
        inv_eff = _apply_moves(_IDENTITY, inv)
        gens.append((inv_eff, inv))
    return gens


# ---------------------------------------------------------------------------
# Table construction (BFS over the LL group) — observable + cached
# ---------------------------------------------------------------------------

_CACHE_PATH = Path(__file__).parent / ".cfop_cache.pkl"
_CACHE_VERSION = 4  # bump when generators / key scheme change


def _build_cross_table() -> dict:
    """BFS from solved over all 18 moves, keyed by cross signature.
    Maps cross_key -> path (solved -> that signature). ~190k states."""
    t0 = time.time()
    table: dict = {_cross_key(_IDENTITY): []}
    queue: deque = deque([(_IDENTITY, [])])
    n = 0
    while queue:
        state, path = queue.popleft()
        n += 1
        if n % 20000 == 0:
            print(f"  [Cross] visited {len(table):>6} states "
                  f"({time.time() - t0:5.1f}s)", flush=True)
        if len(path) >= 9:  # cross is always solvable in <= 8 HTM
            continue
        for mi in range(18):
            ns = _compose(state, _MOVES_PY[mi])
            k = _cross_key(ns)
            if k in table:
                continue
            table[k] = path + [mi]
            queue.append((ns, path + [mi]))
    print(f"  [Cross] done: {len(table)} states ({time.time() - t0:.1f}s)",
          flush=True)
    return table


def _bfs_ll(sources: list[tuple], gens, label: str) -> dict:
    """Multi-source BFS over the LL group.
    Returns {ll_key: path_from_source} for every reachable LL state."""
    t0 = time.time()
    table: dict = {}
    queue: deque = deque()
    for src in sources:
        k = _ll_key(src)
        if k not in table:
            table[k] = []
            queue.append((src, []))
    n = 0
    while queue:
        state, path = queue.popleft()
        n += 1
        if n % 5000 == 0:
            print(f"  [{label}] visited {len(table):>6} states "
                  f"({time.time() - t0:5.1f}s)", flush=True)
        for eff, seq in gens:
            ns = _compose(state, eff)
            k = _ll_key(ns)
            if k in table:
                continue
            table[k] = path + seq
            queue.append((ns, path + seq))
    print(f"  [{label}] done: {len(table)} states ({time.time() - t0:.1f}s)",
          flush=True)
    return table


def _build_tables() -> dict:
    """Build Cross, OLL, and full-LL solution tables."""
    print("cfop: building solver tables (one-time; cached afterward)...",
          flush=True)

    cross = _build_cross_table()

    gens = _build_generators()
    print(f"  generators: {len(_GENERATORS_NOTATION)} algs verified "
          f"F2L-preserving", flush=True)

    # Full-LL table: BFS from solved -> path(solved -> state).
    full = _bfs_ll([_IDENTITY], gens, "LL-full")

    # OLL sources = every reachable LL state that is already oriented.
    oll_sources = [_ll_state_from_key(k) for k in full
                   if oll_solved(_ll_state_from_key(k))]
    oll = _bfs_ll(oll_sources, gens, "OLL")

    return {"version": _CACHE_VERSION, "cross": cross, "full": full, "oll": oll}


def _load_tables() -> dict:
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            if data.get("version") == _CACHE_VERSION:
                print(f"cfop: loaded tables from cache "
                      f"(cross {len(data['cross'])}, LL {len(data['full'])})",
                      flush=True)
                return data
        except Exception as e:  # noqa: BLE001
            print(f"cfop: cache load failed ({e}); rebuilding", flush=True)
    data = _build_tables()
    try:
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"cfop: cached tables to {_CACHE_PATH.name}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"cfop: cache write failed ({e})", flush=True)
    return data


_TABLES = _load_tables()
_CROSS_TABLE: dict = _TABLES["cross"]
_LL_FULL: dict = _TABLES["full"]
_OLL_TABLE: dict = _TABLES["oll"]


# ---------------------------------------------------------------------------
# Per-instance BFS (Cross, F2L)
# ---------------------------------------------------------------------------

def _bfs(start: tuple, goal_fn, move_set: list[int], max_depth: int) -> list[int] | None:
    if goal_fn(start):
        return []
    queue: deque = deque([(start, [])])
    visited: set = {_state_key(start)}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for mi in move_set:
            ns = _compose(state, _MOVES_PY[mi])
            key = _state_key(ns)
            if key in visited:
                continue
            visited.add(key)
            npath = path + [mi]
            if goal_fn(ns):
                return npath
            queue.append((ns, npath))
    return None


# ---------------------------------------------------------------------------
# Stage solvers
# ---------------------------------------------------------------------------

def _solve_cross(state: tuple) -> list[int]:
    path = _CROSS_TABLE.get(_cross_key(state))
    if path is None:
        raise RuntimeError("Cross: signature not in table")
    return _invert_seq(path)


# Corner / edge slots actually moved by each face (read off the MOVES perms).
_FACE_CORNERS = {0: {0, 1, 2, 3}, 1: {4, 5, 6, 7}, 2: {0, 3, 4, 7},
                 3: {1, 2, 5, 6}, 4: {2, 3, 6, 7}, 5: {0, 1, 4, 5}}
_FACE_EDGES = {0: {4, 5, 6, 7}, 1: {8, 9, 10, 11}, 2: {0, 3, 7, 11},
               3: {1, 2, 5, 9}, 4: {2, 3, 6, 10}, 5: {0, 1, 4, 8}}

# For each pair, the corner/edge slots reachable by its restricted face set.
_SLOT_FACES = {0: (0, 3, 4), 1: (0, 2, 4), 2: (0, 3, 5), 3: (0, 2, 5)}  # FR FL BR BL
_REACH_CORNERS = {pi: set().union(*(_FACE_CORNERS[f] for f in faces))
                  for pi, faces in _SLOT_FACES.items()}
_REACH_EDGES = {pi: set().union(*(_FACE_EDGES[f] for f in faces))
                for pi, faces in _SLOT_FACES.items()}


def _solvable_now(state: tuple, pi: int) -> bool:
    """True if both of pair pi's pieces currently sit in slots reachable by the
    slot's restricted move set — a correct necessary condition for inserting
    them without the broader fallback search."""
    cp, _, ep, _ = state
    c, e = _F2L_PAIRS[pi]
    return cp.index(c) in _REACH_CORNERS[pi] and ep.index(e) in _REACH_EDGES[pi]


def _f2l_goal(solved: frozenset, pi: int):
    def goal(s):
        return (cross_solved(s)
                and all(_f2l_pair_solved(s, j) for j in solved)
                and _f2l_pair_solved(s, pi))
    return goal


def _build_pdbs() -> dict:
    """Per-pair pattern database: exact distance to insert the pair tracking
    ONLY its two pieces (corner slot+twist, edge slot+flip) under the restricted
    move set. Other pieces are ignored, so the real (constrained) F2L distance is
    >= this — an admissible heuristic for IDA*. ~hundreds of entries each."""
    pdbs: dict = {}
    for pi in range(4):
        c, e = _F2L_PAIRS[pi]
        moves = _SLOT_MOVES[pi]

        def proj(s):
            cp, ct, ep, ef = s
            cs = cp.index(c)
            es = ep.index(e)
            return (cs, ct[cs], es, ef[es])

        dist = {proj(_IDENTITY): 0}
        queue: deque = deque([(_IDENTITY, 0)])
        while queue:
            st, d = queue.popleft()
            for mi in moves:
                ns = _compose(st, _MOVES_PY[mi])
                p = proj(ns)
                if p not in dist:
                    dist[p] = d + 1
                    queue.append((ns, d + 1))
        pdbs[pi] = dist
    return pdbs


_PDB = _build_pdbs()


def _ida_pair(state: tuple, sv: frozenset, pi: int, max_bound: int = 24) -> list[int] | None:
    """IDA* insertion of pair pi (restricted moves) guided by the pair PDB."""
    c, e = _F2L_PAIRS[pi]
    pdb = _PDB[pi]
    moves = _SLOT_MOVES[pi]
    goal = _f2l_goal(sv, pi)

    def h(s):
        cp, ct, ep, ef = s
        cs = cp.index(c)
        es = ep.index(e)
        return pdb.get((cs, ct[cs], es, ef[es]), 0)

    def dfs(s, g, bound, last, path):
        f = g + h(s)
        if f > bound:
            return f
        if goal(s):
            return True
        best = None
        for mi in moves:
            if last >= 0 and mi // 3 == last // 3:   # no consecutive same-face
                continue
            ns = _compose(s, _MOVES_PY[mi])
            path.append(mi)
            t = dfs(ns, g + 1, bound, mi, path)
            if t is True:
                return True
            if best is None or t < best:
                best = t
            path.pop()
        return best if best is not None else float("inf")

    bound = h(state)
    while bound <= max_bound:
        path: list[int] = []
        t = dfs(state, 0, bound, -1, path)
        if t is True:
            return path
        if t == float("inf"):
            return None
        bound = t
    return None


def _solve_f2l(state: tuple) -> list[int]:
    """Solve the 4 pairs in dynamic order: insert whichever pair is currently
    accessible (shortest first) via PDB-guided IDA*; fall back to a short
    deadlock-break maneuver only when no pair is directly accessible."""
    all_moves: list[int] = []
    solved: set[int] = set()
    deadlock_breaks = 0
    while len(solved) < 4:
        sv = frozenset(solved)
        best = None  # (n_moves, pi, moves)
        for pi in range(4):
            if pi in solved or not _solvable_now(state, pi):
                continue
            mv = _ida_pair(state, sv, pi)
            if mv is not None and (best is None or len(mv) < best[0]):
                best = (len(mv), pi, mv)
        if best is not None:
            _, pi, mv = best
            state = _apply_moves(state, mv)
            all_moves.extend(mv)
            solved.add(pi)
            continue

        # Deadlock: no remaining pair is directly accessible. Search a short
        # maneuver (keeping cross + solved pairs) that makes some buried pair
        # accessible again, then retry. Bounded depth keeps this cheap.
        deadlock_breaks += 1
        if deadlock_breaks > 8:
            raise RuntimeError("F2L stuck (too many deadlock breaks)")

        def unstick(s, sv=sv):
            return (cross_solved(s)
                    and all(_f2l_pair_solved(s, j) for j in sv)
                    and any(_solvable_now(s, pi) for pi in range(4) if pi not in sv))

        mv = _bfs(state, unstick, _F2L_MOVES, max_depth=6)
        if mv is None:
            raise RuntimeError("F2L stuck (no deadlock-break maneuver)")
        state = _apply_moves(state, mv)
        all_moves.extend(mv)
    return all_moves


def _solve_oll(state: tuple) -> list[int]:
    path = _OLL_TABLE.get(_ll_key(state))
    if path is None:
        raise RuntimeError("OLL: LL state not in table (generators incomplete?)")
    return _invert_seq(path)


def _solve_pll(state: tuple) -> list[int]:
    path = _LL_FULL.get(_ll_key(state))
    if path is None:
        raise RuntimeError("PLL: LL state not in table (generators incomplete?)")
    return _invert_seq(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(state: tuple, verbose: bool = False) -> list[int]:
    """Solve a cube state via CFOP. Returns a list of move indices 0..17."""
    solution: list[int] = []

    def stage(name, fn):
        nonlocal state
        moves = fn(state)
        state = _apply_moves(state, moves)
        solution.extend(moves)
        if verbose:
            print(f"  {name:5}: {len(moves):3} moves (total {len(solution)})")

    stage("Cross", _solve_cross)
    stage("F2L", _solve_f2l)
    stage("OLL", _solve_oll)
    stage("PLL", _solve_pll)

    if not cube_solved(state):
        raise RuntimeError("CFOP failed: cube not solved after all stages")
    return solution


def solve_stages(state: tuple) -> dict:
    """Solve and return per-stage move lists (for stage-labeled BC data)."""
    out: dict = {}
    m = _solve_cross(state); state = _apply_moves(state, m); out["cross"] = m
    m = _solve_f2l(state);   state = _apply_moves(state, m); out["f2l"] = m
    m = _solve_oll(state);   state = _apply_moves(state, m); out["oll"] = m
    m = _solve_pll(state);   state = _apply_moves(state, m); out["pll"] = m
    assert cube_solved(state)
    return out


if __name__ == "__main__":
    import random
    random.seed(0)
    n_test = 20
    print(f"\nsolving {n_test} random 25-move scrambles...")
    t0 = time.time()
    lengths = []
    for i in range(n_test):
        s = _IDENTITY
        for _ in range(25):
            s = _compose(s, _MOVES_PY[random.randrange(18)])
        sol = solve(s)
        assert cube_solved(_apply_moves(s, sol)), f"scramble {i} not solved!"
        lengths.append(len(sol))
        print(f"  scramble {i:2}: {len(sol):3} moves  "
              f"({(i + 1) / (time.time() - t0):.1f}/s)", flush=True)
    print(f"\nall {n_test} solved. "
          f"avg {sum(lengths) / len(lengths):.1f}, max {max(lengths)} moves")
