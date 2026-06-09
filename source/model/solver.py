"""Goal-conditioned cube move predictor (MLX).

Architecture
------------
Input:  goal state + current state, each encoded as 20 piece tokens.
        Concatenated to a 40-token sequence.
Output: logits over 18 moves (6 faces × {1-turn, 2-turn, 3-turn}).

Each piece token = slot_emb + role_emb + piece_id_emb + orientation_emb.
A Transformer encoder attends over all 40 tokens; mean-pooled output feeds
the 18-class head.

Noise level ``t`` (diffusion timestep) is optionally broadcast-added to every
token so the same weights serve both supervised and diffusion training.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

N_MOVES = 18   # 6 faces × {1, 2, 3} turns
T_MAX = 100    # max diffusion timesteps


class _Layer(nn.Module):
    """Pre-norm Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, n_heads, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ffn_dim)
        self.ff2 = nn.Linear(ffn_dim, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)
        h = self.norm2(x)
        x = x + self.ff2(nn.gelu(self.ff1(h)))
        return x


class CubeSolver(nn.Module):
    """
    Parameters
    ----------
    d_model  : embedding / hidden dimension
    n_layers : number of Transformer encoder layers
    n_heads  : attention heads (must divide d_model)
    ffn_mult : FFN hidden dim = d_model × ffn_mult
    t_max    : maximum noise timestep (for diffusion use)

    Forward signature
    -----------------
    model(goal, curr, t=None) -> logits [B, 18]

    goal / curr : tuple (cp, ct, ep, ef) of int32 arrays
        cp  [B, 8]  corner positions (piece id at each slot)
        ct  [B, 8]  corner twists   (0 / 1 / 2)
        ep  [B, 12] edge positions
        ef  [B, 12] edge flips      (0 / 1)
    t   : [B] int32 noise level (optional)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        t_max: int = T_MAX,
    ):
        super().__init__()
        self.slot_emb = nn.Embedding(20, d_model)   # slot index 0-19 within a state
        self.role_emb = nn.Embedding(2, d_model)    # 0 = goal, 1 = current
        self.cp_emb   = nn.Embedding(8, d_model)    # corner piece id
        self.ct_emb   = nn.Embedding(3, d_model)    # corner twist 0/1/2
        self.ep_emb   = nn.Embedding(12, d_model)   # edge piece id
        self.ef_emb   = nn.Embedding(2, d_model)    # edge flip 0/1
        self.t_emb    = nn.Embedding(t_max + 1, d_model)

        ffn_dim = d_model * ffn_mult
        self.layers = [_Layer(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_MOVES)

    # ------------------------------------------------------------------ encode
    def _encode(
        self,
        cp: mx.array,   # [B, 8]
        ct: mx.array,   # [B, 8]
        ep: mx.array,   # [B, 12]
        ef: mx.array,   # [B, 12]
        role: int,
    ) -> mx.array:      # [B, 20, d_model]
        rv = self.role_emb(mx.array([role]))   # [1, d]

        # corner tokens: slots 0-7
        c = (self.slot_emb(mx.arange(8))[None]   # [1, 8, d]
             + rv[:, None, :]                      # [1, 1, d]
             + self.cp_emb(cp)                    # [B, 8, d]
             + self.ct_emb(ct))                   # [B, 8, d]

        # edge tokens: slots 8-19
        e = (self.slot_emb(mx.arange(8, 20))[None]
             + rv[:, None, :]
             + self.ep_emb(ep)
             + self.ef_emb(ef))

        return mx.concatenate([c, e], axis=1)   # [B, 20, d]

    # ----------------------------------------------------------------- forward
    def __call__(
        self,
        goal: tuple,
        curr: tuple,
        t: mx.array | None = None,
    ) -> mx.array:  # [B, N_MOVES]
        g = self._encode(*goal, role=0)             # [B, 20, d]
        c = self._encode(*curr, role=1)             # [B, 20, d]
        x = mx.concatenate([g, c], axis=1)          # [B, 40, d]

        if t is not None:
            x = x + self.t_emb(t)[:, None, :]      # [B, 1, d] broadcast

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x.mean(axis=1))            # [B, N_MOVES]

    def n_params(self) -> int:
        return sum(p.size for _, p in tree_flatten(self.parameters()))
