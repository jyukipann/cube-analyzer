"""state_to_net is validated against the independent facelet oracle."""
import random

import pytest

from oracle import oracle_net
from state import MOVES, State
from vis_util import state_to_net

MOVE_NAMES = ['U', 'D', 'L', 'R', 'F', 'B']


def net_list(state):
    return state_to_net(state).tolist()


def apply_seq(seq):
    s = State()
    for name in seq:
        s = s @ MOVES[name]
    return s


def test_solved_net_is_identity():
    net = net_list(State())
    for face in range(6):
        assert net[face] == [[face] * 3] * 3


@pytest.mark.parametrize("name", MOVE_NAMES)
def test_single_move_matches_oracle(name):
    assert net_list(MOVES[name]) == oracle_net([name])


def test_r_move_known_expected():
    F, R, L, B, U, D = 0, 1, 2, 3, 4, 5
    expected = [
        [[F, F, D], [F, F, D], [F, F, D]],
        [[R, R, R], [R, R, R], [R, R, R]],
        [[L, L, L], [L, L, L], [L, L, L]],
        [[U, B, B], [U, B, B], [U, B, B]],
        [[U, U, F], [U, U, F], [U, U, F]],
        [[D, D, B], [D, D, B], [D, D, B]],
    ]
    assert net_list(MOVES['R']) == expected


@pytest.mark.parametrize("seed", range(50))
def test_random_scramble_matches_oracle(seed):
    rng = random.Random(seed)
    seq = [rng.choice(MOVE_NAMES) for _ in range(rng.randint(1, 30))]
    assert net_list(apply_seq(seq)) == oracle_net(seq)


def test_every_sticker_count_preserved():
    """Each color appears exactly 9 times in any reachable net."""
    rng = random.Random(123)
    seq = [rng.choice(MOVE_NAMES) for _ in range(40)]
    net = net_list(apply_seq(seq))
    counts = [0] * 6
    for face in net:
        for row in face:
            for c in row:
                counts[c] += 1
    assert counts == [9] * 6
