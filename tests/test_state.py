"""Group algebra of State: identity, inverse, order, composition."""
import random

import pytest

from state import MOVES, State

MOVE_NAMES = ['U', 'D', 'L', 'R', 'F', 'B']


def apply_seq(seq):
    s = State()
    for name in seq:
        s = s @ MOVES[name]
    return s


def test_identity_is_neutral():
    s = State()
    for m in MOVES.values():
        assert s @ m == m
        assert m @ s == m


@pytest.mark.parametrize("name", MOVE_NAMES)
def test_inverse(name):
    m = MOVES[name]
    s = State()
    assert m @ (~m) == s
    assert (~m) @ m == s


@pytest.mark.parametrize("name", MOVE_NAMES)
def test_quarter_turn_order_four(name):
    m = MOVES[name]
    assert 4 * m == State()
    assert 2 * m == m @ m
    assert 3 * m == ~m  # a quarter turn cubed is its inverse


def test_negative_multiplication():
    r = MOVES['R']
    assert -1 * r == ~r
    assert -2 * r == ~(2 * r)


def test_sexy_move_order_six():
    # (R U R' U') has order 6
    seq = ['R', 'U', "R'", "U'"]

    def rurupu():
        return MOVES['R'] @ MOVES['U'] @ (~MOVES['R']) @ (~MOVES['U'])

    acc = State()
    for _ in range(6):
        acc = acc @ rurupu()
    assert acc == State()


def test_commutator_identity():
    # [R, U] = R U R' U' is not identity, but applied... just sanity it differs
    r, u = MOVES['R'], MOVES['U']
    assert (r @ u) != (u @ r)


def test_scramble_then_inverse_solves():
    rng = random.Random(7)
    names = [rng.choice(MOVE_NAMES) for _ in range(25)]
    scramble = apply_seq(names)
    assert scramble @ (~scramble) == State()
    # undo move-by-move in reverse
    undo = State()
    for name in reversed(names):
        undo = undo @ (~MOVES[name])
    assert scramble @ undo == State()


def test_associativity():
    rng = random.Random(11)
    a = apply_seq([rng.choice(MOVE_NAMES) for _ in range(5)])
    b = apply_seq([rng.choice(MOVE_NAMES) for _ in range(5)])
    c = apply_seq([rng.choice(MOVE_NAMES) for _ in range(5)])
    assert (a @ b) @ c == a @ (b @ c)


def test_mutable_compose_matches_immutable():
    a = apply_seq(['R', 'U', 'F'])
    b = MOVES['L']
    immut = a @ b
    a @= b
    assert a == immut
