"""Tests for value-iteration batch generation (generate_batch_value_iter)."""
import random

import data
from data import (
    generate_batch_value_iter,
    _compose,
    _MOVES_PY,
    _IDENTITY,
    _INV_IDX,
)


def test_children_match_compose():
    """The stored 18 children must equal current . a for each move a."""
    random.seed(0)
    b = generate_batch_value_iter(64, t_cap=26, identity_goal_frac=0.5)

    ccp = b['ccp'].tolist(); cct = b['cct'].tolist()
    cep = b['cep'].tolist(); cef = b['cef'].tolist()
    chcp = b['chcp'].tolist(); chct = b['chct'].tolist()
    chep = b['chep'].tolist(); chef = b['chef'].tolist()

    for i in range(64):
        current = (ccp[i], cct[i], cep[i], cef[i])
        for a in range(18):
            child = _compose(current, _MOVES_PY[a])
            assert chcp[i][a] == child[0]
            assert chct[i][a] == child[1]
            assert chep[i][a] == child[2]
            assert chef[i][a] == child[3]


def test_child_is_goal_mask_correct():
    """child_is_goal[i,a] == 1 iff current . a equals the goal state."""
    random.seed(1)
    b = generate_batch_value_iter(128, t_cap=26, identity_goal_frac=0.5)

    gcp = b['gcp'].tolist(); gct = b['gct'].tolist()
    gep = b['gep'].tolist(); gef = b['gef'].tolist()
    chcp = b['chcp'].tolist(); chct = b['chct'].tolist()
    chep = b['chep'].tolist(); chef = b['chef'].tolist()
    mask = b['child_is_goal'].tolist()

    for i in range(128):
        goal = (gcp[i], gct[i], gep[i], gef[i])
        for a in range(18):
            child = (chcp[i][a], chct[i][a], chep[i][a], chef[i][a])
            expected = 1 if list(child) == [goal[0], goal[1], goal[2], goal[3]] else 0
            assert mask[i][a] == expected


def test_distance_one_has_exactly_one_goal_child():
    """A current state exactly one move from the goal (t==1 with the policy
    target undoing that move) has at least one goal-child, and applying the
    policy target reaches the goal."""
    random.seed(2)
    b = generate_batch_value_iter(256, t_cap=26, identity_goal_frac=0.5)

    gcp = b['gcp'].tolist(); gct = b['gct'].tolist()
    gep = b['gep'].tolist(); gef = b['gef'].tolist()
    ccp = b['ccp'].tolist(); cct = b['cct'].tolist()
    cep = b['cep'].tolist(); cef = b['cef'].tolist()
    t = b['t'].tolist(); target = b['target'].tolist()
    mask = b['child_is_goal'].tolist()

    n_t1 = 0
    for i in range(256):
        if t[i] != 1:
            continue
        n_t1 += 1
        goal = (gcp[i], gct[i], gep[i], gef[i])
        current = (ccp[i], cct[i], cep[i], cef[i])
        # The policy target move must take current to the goal.
        reached = _compose(current, _MOVES_PY[target[i]])
        assert list(reached) == [goal[0], goal[1], goal[2], goal[3]]
        # Therefore at least one child equals the goal.
        assert sum(mask[i]) >= 1
    assert n_t1 > 0  # sanity: some t==1 samples exist
