"""Tests for the CFOP solver (source/cfop.py)."""
import random

import pytest

from cfop import (
    _IDENTITY,
    _MOVES_PY,
    _apply_moves,
    _compose,
    cross_solved,
    cube_solved,
    f2l_solved,
    oll_solved,
    solve,
    solve_stages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SCRAMBLES = 15
_SCRAMBLE_LEN = 25
_SEED = 42


def _make_scrambles(n: int = _N_SCRAMBLES, length: int = _SCRAMBLE_LEN, seed: int = _SEED):
    """Return a list of *n* random cube states, each scrambled with *length* moves."""
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        s = _IDENTITY
        for _ in range(length):
            s = _compose(s, _MOVES_PY[rng.randrange(18)])
        states.append(s)
    return states


_SCRAMBLES = _make_scrambles()


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

def test_solve_identity():
    """Solving the already-solved cube must keep it solved."""
    solution = solve(_IDENTITY)
    result = _apply_moves(_IDENTITY, solution)
    assert cube_solved(result), "After solving identity the cube must still be solved"


# ---------------------------------------------------------------------------
# 2. Random scrambles — correctness
# ---------------------------------------------------------------------------

def test_solve_random_scrambles():
    """solve() must produce a move list that yields a solved cube for every scramble."""
    for i, state in enumerate(_SCRAMBLES):
        solution = solve(state)
        assert isinstance(solution, list), f"scramble {i}: solve() did not return a list"
        result = _apply_moves(state, solution)
        assert cube_solved(result), f"scramble {i}: cube not solved after applying solution"


# ---------------------------------------------------------------------------
# 3. Reasonable move count
# ---------------------------------------------------------------------------

def test_solution_move_count_reasonable():
    """Every solution should be no longer than 120 moves (typical CFOP: ~90)."""
    for i, state in enumerate(_SCRAMBLES):
        solution = solve(state)
        assert len(solution) <= 120, (
            f"scramble {i}: solution length {len(solution)} exceeds 120"
        )


# ---------------------------------------------------------------------------
# 4. All move indices in range
# ---------------------------------------------------------------------------

def test_solution_moves_in_range():
    """Every move index returned by solve() must be in 0..17."""
    for i, state in enumerate(_SCRAMBLES):
        solution = solve(state)
        for j, mi in enumerate(solution):
            assert 0 <= mi <= 17, (
                f"scramble {i}, position {j}: move index {mi} out of range 0..17"
            )


# ---------------------------------------------------------------------------
# 5. Stage goals progress cumulatively
# ---------------------------------------------------------------------------

def test_stage_goals_progress():
    """Each CFOP stage must achieve its goal without breaking previous stages."""
    for i, scrambled in enumerate(_SCRAMBLES[:10]):
        stages = solve_stages(scrambled)

        # Apply cross moves and check cross goal
        s = scrambled
        s = _apply_moves(s, stages["cross"])
        assert cross_solved(s), f"scramble {i}: cross not solved after cross stage"

        # Apply F2L moves and check F2L goal (cross must still hold)
        s = _apply_moves(s, stages["f2l"])
        assert cross_solved(s), f"scramble {i}: cross broken after f2l stage"
        assert f2l_solved(s), f"scramble {i}: f2l not solved after f2l stage"

        # Apply OLL moves and check OLL goal (f2l must still hold)
        s = _apply_moves(s, stages["oll"])
        assert f2l_solved(s), f"scramble {i}: f2l broken after oll stage"
        assert oll_solved(s), f"scramble {i}: oll not solved after oll stage"

        # Apply PLL moves and check fully solved
        s = _apply_moves(s, stages["pll"])
        assert cube_solved(s), f"scramble {i}: cube not solved after pll stage"


# ---------------------------------------------------------------------------
# 6. Concatenated stage moves also solve the cube
# ---------------------------------------------------------------------------

def test_stages_concat_equals_solve_length_or_solves():
    """The concatenation of all four solve_stages lists must solve the cube."""
    for i, scrambled in enumerate(_SCRAMBLES[:10]):
        stages = solve_stages(scrambled)
        full_solution = stages["cross"] + stages["f2l"] + stages["oll"] + stages["pll"]
        result = _apply_moves(scrambled, full_solution)
        assert cube_solved(result), (
            f"scramble {i}: concatenated stage moves do not yield a solved cube"
        )
