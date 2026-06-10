"""Tests for CubeSolver and data generation."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from data import _IDENTITY, _INV_IDX, _MOVES_PY, _compose, generate_batch
from model.solver import N_MOVES, CubeSolver


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------

def test_generate_batch_shapes():
    batch = generate_batch(8, t_max=10)
    assert batch['gcp'].shape == (8, 8)
    assert batch['gct'].shape == (8, 8)
    assert batch['gep'].shape == (8, 12)
    assert batch['gef'].shape == (8, 12)
    assert batch['ccp'].shape == (8, 8)
    assert batch['cct'].shape == (8, 8)
    assert batch['cep'].shape == (8, 12)
    assert batch['cef'].shape == (8, 12)
    assert batch['t'].shape == (8,)
    assert batch['target'].shape == (8,)


def test_target_in_range():
    batch = generate_batch(256, t_max=100)
    targets = batch['target'].tolist()
    assert all(0 <= x < N_MOVES for x in targets)


def test_t_in_range():
    batch = generate_batch(256, t_max=50)
    ts = batch['t'].tolist()
    assert all(1 <= x <= 50 for x in ts)


def test_inverse_idx_all_moves():
    """move ∘ inverse(move) == identity for all 18 moves."""
    for idx in range(18):
        state = _compose(_IDENTITY, _MOVES_PY[idx])
        state = _compose(state, _MOVES_PY[_INV_IDX[idx]])
        assert state == _IDENTITY, f"move {idx}: compose with inverse != identity"


def test_compose_order_four_quarter_turns():
    """Each of the 6 quarter-turn moves has order 4."""
    for fi in range(6):
        idx = fi * 3   # turns=1 slot
        state = _IDENTITY
        for _ in range(4):
            state = _compose(state, _MOVES_PY[idx])
        assert state == _IDENTITY, f"move {idx}: order != 4"


def test_compose_order_two_half_turns():
    """Each half-turn (turns=2) has order 2."""
    for fi in range(6):
        idx = fi * 3 + 1   # turns=2 slot
        state = _compose(_IDENTITY, _MOVES_PY[idx])
        state = _compose(state, _MOVES_PY[idx])
        assert state == _IDENTITY, f"move {idx}: half-turn order != 2"


# ---------------------------------------------------------------------------
# model forward pass
# ---------------------------------------------------------------------------

def _tiny_model():
    return CubeSolver(d_model=64, n_layers=2, n_heads=4, ffn_mult=2)


def _batch(B=4):
    return generate_batch(B, t_max=10)


def test_forward_shape_with_t():
    model = _tiny_model()
    batch = _batch(4)
    logits = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        t=batch['t'],
    )
    assert logits.shape == (4, N_MOVES)


def test_forward_shape_without_t():
    model = _tiny_model()
    batch = _batch(4)
    logits = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
    )
    assert logits.shape == (4, N_MOVES)


def test_loss_is_finite():
    model = _tiny_model()
    batch = _batch(32)
    logits = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        t=batch['t'],
    )
    loss = nn.losses.cross_entropy(logits, batch['target']).mean()
    mx.eval(loss)
    v = float(loss)
    assert v > 0
    assert v < 10   # untrained loss ≈ log(18) ≈ 2.89


def test_initial_loss_near_uniform():
    """Untrained model should produce near-uniform logits → loss ≈ log(18)."""
    import math
    model = _tiny_model()
    batch = generate_batch(512, t_max=100)
    logits = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        t=batch['t'],
    )
    loss = float(nn.losses.cross_entropy(logits, batch['target']).mean())
    mx.eval()
    assert abs(loss - math.log(N_MOVES)) < 1.5, f"initial loss {loss:.3f} far from log(18)={math.log(N_MOVES):.3f}"


# ---------------------------------------------------------------------------
# parameter counts
# ---------------------------------------------------------------------------

def test_n_params_small():
    model = CubeSolver(d_model=128, n_layers=4, n_heads=4)
    n = model.n_params()
    assert 700_000 < n < 1_200_000, f"small param count unexpected: {n:,}"


def test_n_params_medium():
    model = CubeSolver(d_model=256, n_layers=6, n_heads=8)
    n = model.n_params()
    assert 4_000_000 < n < 7_000_000, f"medium param count unexpected: {n:,}"


# ---------------------------------------------------------------------------
# goal-conditioning: same current, different goal → different logits
# ---------------------------------------------------------------------------

def test_goal_conditioning():
    model = CubeSolver(d_model=64, n_layers=2, n_heads=4, ffn_mult=2)
    batch = _batch(4)

    # Swap goal and current → logits should differ
    logits_a = model(
        goal=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        curr=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        t=batch['t'],
    )
    logits_b = model(
        goal=(batch['ccp'], batch['cct'], batch['cep'], batch['cef']),
        curr=(batch['gcp'], batch['gct'], batch['gep'], batch['gef']),
        t=batch['t'],
    )
    mx.eval(logits_a, logits_b)
    assert not mx.all(logits_a == logits_b)
