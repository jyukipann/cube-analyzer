"""Tests for the LLM cube tool kernel (cube_tools.py)."""
import random

import pytest

import cube_tools as ct
from cube_tools import (
    CubeSession, parse_moves, invert_moves, NOTATION, MoveParseError,
)


def test_notation_roundtrip():
    assert len(NOTATION) == 18
    for idx, tok in enumerate(NOTATION):
        assert parse_moves(tok) == [idx]
    # lowercase + unicode prime normalize
    assert parse_moves("r u r’ u'") == parse_moves("R U R' U'")


def test_parse_error_names_token():
    with pytest.raises(MoveParseError):
        parse_moves("R X U")


def test_inverse_undoes():
    s = CubeSession(seed=1)
    seq = "R U2 F' L D' B"
    s.apply(seq)
    s.apply(invert_moves(seq))
    assert s.observe()["is_solved"]


def test_scramble_then_inverse_solves():
    s = CubeSession(seed=7)
    obs = s.scramble(12)
    assert not obs["is_solved"]
    undo = invert_moves(' '.join(NOTATION[i] for i in s.scramble_moves))
    s.apply(undo)
    assert s.observe()["is_solved"]


def test_simulate_does_not_commit():
    s = CubeSession(seed=2)
    s.scramble(6)
    before = s.observe()["pieces_solved"]
    sim = s.simulate("R U R' U'")
    assert sim["committed"] is False
    assert "delta_pieces_solved" in sim
    # real state unchanged
    assert s.observe()["pieces_solved"] == before


def test_solved_counts_full_on_solved():
    s = CubeSession(seed=0)
    o = s.observe()
    assert o["corners_solved"] == 8
    assert o["edges_solved"] == 12
    assert o["pieces_solved"] == 20
    assert o["is_solved"]


def test_scramble_avoids_same_face_repeats():
    s = CubeSession(seed=3)
    s.scramble(30)
    faces = [i // 3 for i in s.scramble_moves]
    assert all(faces[k] != faces[k + 1] for k in range(len(faces) - 1))
