"""Group-theoretic Rubik's cube State (MLX backend).

A ``State`` is simultaneously a cube configuration and a move: both are
elements of the cube group, so the same class represents states and the
operators that act on them.

Encoding (all indexed by *slot* / position):
    corner_positions    : int8[8]      piece id sitting at each corner slot
    corner_orientations : float16[8,2] [cos t, sin t] twist of that piece
    edge_positions      : int8[12]     piece id sitting at each edge slot
    edge_orientations   : int8[12]     flip as +1 / -1

Group operators:
    a @ b   immutable compose: apply b to a
    a @= b  mutable compose
    ~a      inverse element
    n * a   repeat n times (negative = inverse direction)
    a == b  equality via integer twist / flip
    State() identity (solved cube)

Composition is the wreath-product action: applying ``b`` permutes ``a`` by
``b``'s permutation *and* carries ``a``'s orientation along that permutation
before adding ``b``'s twist. The numeric MOVES tables are derived from the 3D
geometry in ``tests`` and verified against an independent facelet oracle.
"""

import math
import typing
from math import pi

import mlx.core as mx

# twist index 0/1/2 -> unit vector [cos, sin] (rotation by 0, 120, 240 deg)
TWIST_TABLE = mx.array([
    [math.cos(0.0), math.sin(0.0)],          # 0
    [math.cos(2 / 3 * pi), math.sin(2 / 3 * pi)],  # 1
    [math.cos(4 / 3 * pi), math.sin(4 / 3 * pi)],  # 2
], dtype=mx.float16)


def inverse_permutation(p: mx.array) -> mx.array:
    """Inverse of a permutation given as an index array."""
    # argsort of a permutation of 0..n-1 is exactly its inverse permutation.
    return mx.argsort(p.astype(mx.int32)).astype(p.dtype)


def get_twist_vec(twist: int) -> list[float]:
    """twist index -> [cos, sin] orientation vector."""
    return TWIST_TABLE[twist % 3].tolist()


class State:
    """Cube state == cube move == cube-group element."""

    def __init__(
        self,
        corner_positions: typing.Optional[mx.array | list[int]] = None,
        corner_orientations: typing.Optional[mx.array | list] = None,
        edge_positions: typing.Optional[mx.array | list[int]] = None,
        edge_orientations: typing.Optional[mx.array | list[int]] = None,
    ):
        none_all = (
            corner_positions is None and corner_orientations is None
            and edge_positions is None and edge_orientations is None)
        not_none_all = (
            corner_positions is not None and corner_orientations is not None
            and edge_positions is not None and edge_orientations is not None)

        if none_all:
            self.corner_positions = mx.arange(8, dtype=mx.int8)
            self.corner_orientations = mx.tile(
                mx.array([[1.0, 0.0]], dtype=mx.float16), (8, 1))
            self.edge_positions = mx.arange(12, dtype=mx.int8)
            self.edge_orientations = mx.ones(12, dtype=mx.int8)
        elif not_none_all:
            self.corner_positions = _as_int8(corner_positions)
            self.corner_orientations = _as_corner_orientations(corner_orientations)
            self.edge_positions = _as_int8(edge_positions)
            self.edge_orientations = _as_int8(edge_orientations)
        else:
            raise ValueError(
                "Either all or none of the state components must be provided.")

    # ------------------------------------------------------------------ utils
    def clone(self) -> "State":
        return State(
            corner_positions=mx.array(self.corner_positions),
            corner_orientations=mx.array(self.corner_orientations),
            edge_positions=mx.array(self.edge_positions),
            edge_orientations=mx.array(self.edge_orientations))

    # ------------------------------------------------------------- group action
    def _corner_apply(self, sp: mx.array, so: mx.array) -> tuple[mx.array, mx.array]:
        idx = sp.astype(mx.int32)
        p = self.corner_positions[idx]
        # carry self orientation along b's permutation, then compose b's twist
        o = self.corner_orientations[idx]
        o = mx.stack([
            o[:, 0] * so[:, 0] - o[:, 1] * so[:, 1],  # cos(a+b)
            o[:, 1] * so[:, 0] + o[:, 0] * so[:, 1],  # sin(a+b)
        ], axis=1)
        return p, o

    def _edge_apply(self, sp: mx.array, so: mx.array) -> tuple[mx.array, mx.array]:
        idx = sp.astype(mx.int32)
        p = self.edge_positions[idx]
        o = self.edge_orientations[idx] * so
        return p, o

    def _apply(self, state: "State"):
        cp, co = self._corner_apply(state.corner_positions, state.corner_orientations)
        ep, eo = self._edge_apply(state.edge_positions, state.edge_orientations)
        return cp, co, ep, eo

    def apply(self, state: "State") -> None:
        """Mutate self by applying ``state``."""
        self.corner_positions, self.corner_orientations, \
            self.edge_positions, self.edge_orientations = self._apply(state)

    def get_applied(self, state: "State") -> "State":
        """Return a new state = self with ``state`` applied (immutable)."""
        cp, co, ep, eo = self._apply(state)
        return State(cp, co, ep, eo)

    # ------------------------------------------------------------------ operators
    def __matmul__(self, state: "State") -> "State":
        return self.get_applied(state)

    def __imatmul__(self, state: "State") -> "State":
        self.apply(state)
        return self

    def __invert__(self) -> "State":
        icp = inverse_permutation(self.corner_positions)
        iep = inverse_permutation(self.edge_positions)
        # ~a.co[i] = conj(a.co[icp[i]]) ; ~a.eo[i] = a.eo[iep[i]]
        conj = self.corner_orientations * mx.array([1.0, -1.0], dtype=mx.float16)
        ico = conj[icp.astype(mx.int32)]
        ieo = self.edge_orientations[iep.astype(mx.int32)]
        return State(icp, ico, iep, ieo)

    def __mul__(self, n: int) -> "State":
        assert isinstance(n, int), "multiplication only defined for integers"
        assert n != 0, "multiplication by 0 is undefined"
        if n < 0:
            return (~self).__mul__(-n)
        result = self.clone()
        for _ in range(n - 1):
            result = result @ self
        return result

    def __rmul__(self, n: int) -> "State":
        return self.__mul__(n)

    def __eq__(self, state: "State") -> bool:
        return (
            bool(mx.all(self.corner_positions == state.corner_positions))
            and bool(mx.all(self.twist_co == state.twist_co))
            and bool(mx.all(self.edge_positions == state.edge_positions))
            and bool(mx.all(self.twist_eo == state.twist_eo)))

    def __hash__(self):
        return hash((
            tuple(self.corner_positions.tolist()),
            tuple(self.twist_co.tolist()),
            tuple(self.edge_positions.tolist()),
            tuple(self.twist_eo.tolist())))

    def __str__(self) -> str:
        return (
            f"corner_positions    {self.corner_positions.tolist()}\n"
            f"corner_orientations {self.twist_co.tolist()}\n"
            f"edge_positions      {self.edge_positions.tolist()}\n"
            f"edge_orientations   {self.twist_eo.tolist()}\n")

    # ------------------------------------------------------------- integer views
    @property
    def twist_co(self) -> mx.array:
        """int8[8] twist index 0/1/2 via nearest TWIST_TABLE vector."""
        co = self.corner_orientations
        norm = mx.maximum(mx.sqrt(mx.sum(co * co, axis=1, keepdims=True)), 1e-4)
        co = co / norm
        return mx.argmax(co @ TWIST_TABLE.T, axis=1).astype(mx.int8)

    @property
    def twist_eo(self) -> mx.array:
        """int8[12] flip 0/1."""
        return ((self.edge_orientations.astype(mx.float16) - 1) / -2).astype(mx.int8)


def _as_int8(x) -> mx.array:
    if isinstance(x, mx.array):
        return x.astype(mx.int8)
    return mx.array(x, dtype=mx.int8)


def _as_corner_orientations(x) -> mx.array:
    """Accept (8,2) [cos,sin] arrays/lists, or (8,) twist indices."""
    if isinstance(x, mx.array):
        return x.astype(mx.float16)
    arr = mx.array(x)
    if arr.ndim == 2 and arr.shape == (8, 2):
        return arr.astype(mx.float16)
    if arr.ndim == 1 and arr.shape[0] == 8:
        return TWIST_TABLE[arr.astype(mx.int32)].astype(mx.float16)
    raise ValueError("corner_orientations must have shape (8, 2) or (8,)")


# Quarter-turn moves (QTM), derived from cube geometry and verified against an
# independent facelet oracle in tests/. corner orientation given as twist index
# 0/1/2; edge orientation given as flip +1/-1.
MOVES = {
    'U': State(
        [3, 0, 1, 2, 4, 5, 6, 7],
        [2, 1, 2, 1, 0, 0, 0, 0],
        [0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
    ),
    'D': State(
        [0, 1, 2, 3, 5, 6, 7, 4],
        [0, 0, 0, 0, 1, 2, 1, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 8],
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
    ),
    'L': State(
        [4, 1, 2, 0, 7, 5, 6, 3],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [11, 1, 2, 7, 4, 5, 6, 0, 8, 9, 10, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
    'R': State(
        [0, 2, 6, 3, 4, 1, 5, 7],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 9, 3, 4, 2, 6, 7, 8, 1, 10, 11],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
    'F': State(
        [0, 1, 3, 7, 4, 5, 2, 6],
        [0, 0, 1, 2, 0, 0, 2, 1],
        [0, 1, 6, 10, 4, 5, 3, 7, 8, 9, 2, 11],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
    'B': State(
        [1, 5, 2, 3, 0, 4, 6, 7],
        [1, 2, 0, 0, 2, 1, 0, 0],
        [4, 8, 2, 3, 1, 5, 6, 7, 0, 9, 10, 11],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
}


if __name__ == "__main__":
    s = State()
    for name, m in MOVES.items():
        assert m @ (~m) == s, f"{name} @ ~{name} != identity"
        assert (~m) @ m == s, f"~{name} @ {name} != identity"
        assert 4 * m == s, f"{name}^4 != identity"
    print("identity, inverse, and order-4 checks passed for all moves")
    print(MOVES['R'])
