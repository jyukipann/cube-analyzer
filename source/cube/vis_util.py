"""State -> net (展開図) rendering (MLX backend).

``state_to_net(state)`` returns an int array ``net[6, 3, 3]`` where each cell
holds a face index (F=0, R=1, L=2, B=3, U=4, D=5).

The mapping is built from a canonical, *consistently oriented* (CCW seen from
outside) facelet ordering of every corner and edge slot. Because the ordering
has uniform chirality across all slots, a single cyclic-offset formula renders
the twist/flip of every piece correctly:

    corner: net[slot.faces[(k + twist) % 3]] = piece.colors[k]
    edge:   net[slot.faces[(k + flip)  % 2]] = piece.colors[k]

``piece.colors`` are simply the canonical faces of the piece's home slot.
``CORNER_FACES_POSITIONS`` / ``EDGE_FACES_POSITIONS`` and this formula are
verified against an independent facelet oracle in tests/.

      U
  L F R B      <- net layout
      D
"""

import mlx.core as mx

from state import State

# face indices
F, R, L, B, U, D = 0, 1, 2, 3, 4, 5
ITOA = {F: 'F', R: 'R', L: 'L', B: 'B', U: 'U', D: 'D'}

# in-face subcube cell positions (row, col)
FP00, FP01, FP02 = (0, 0), (0, 1), (0, 2)
FP10, FP11, FP12 = (1, 0), (1, 1), (1, 2)
FP20, FP21, FP22 = (2, 0), (2, 1), (2, 2)

# corner slot -> canonical CCW ordered (face, cell), one entry per sticker.
CORNER_FACES_POSITIONS = [
    [(L, FP00), (U, FP00), (B, FP02)],  # C00
    [(R, FP02), (B, FP00), (U, FP02)],  # C01
    [(R, FP00), (U, FP22), (F, FP02)],  # C02
    [(L, FP02), (F, FP00), (U, FP20)],  # C03
    [(L, FP20), (B, FP22), (D, FP20)],  # C04
    [(R, FP22), (D, FP22), (B, FP20)],  # C05
    [(R, FP20), (F, FP22), (D, FP02)],  # C06
    [(L, FP22), (D, FP00), (F, FP20)],  # C07
]

# edge slot -> canonical ordered (face, cell), one entry per sticker.
EDGE_FACES_POSITIONS = [
    [(L, FP10), (B, FP12)],  # E00
    [(R, FP12), (B, FP10)],  # E01
    [(R, FP10), (F, FP12)],  # E02
    [(L, FP12), (F, FP10)],  # E03
    [(U, FP01), (B, FP01)],  # E04
    [(R, FP01), (U, FP12)],  # E05
    [(U, FP21), (F, FP01)],  # E06
    [(L, FP01), (U, FP10)],  # E07
    [(D, FP21), (B, FP21)],  # E08
    [(R, FP21), (D, FP12)],  # E09
    [(D, FP01), (F, FP21)],  # E10
    [(L, FP21), (D, FP10)],  # E11
]

# face index each corner/edge sticker belongs to (== its home colors)
CORNER_FACES = [[fc for fc, _ in slot] for slot in CORNER_FACES_POSITIONS]
EDGE_FACES = [[fc for fc, _ in slot] for slot in EDGE_FACES_POSITIONS]


def state_to_net(state: State) -> mx.array:
    """Render a State to an int32 net tensor of shape [6, 3, 3]."""
    net = [[[fc for _ in range(3)] for _ in range(3)] for fc in range(6)]

    twists = state.twist_co.tolist()
    cp = state.corner_positions.tolist()
    for slot in range(8):
        piece = cp[slot]
        twist = twists[slot]
        colors = CORNER_FACES[piece]
        for k in range(3):
            fc, (r, c) = CORNER_FACES_POSITIONS[slot][(k + twist) % 3]
            net[fc][r][c] = colors[k]

    flips = state.twist_eo.tolist()
    ep = state.edge_positions.tolist()
    for slot in range(12):
        piece = ep[slot]
        flip = flips[slot]
        colors = EDGE_FACES[piece]
        for k in range(2):
            fc, (r, c) = EDGE_FACES_POSITIONS[slot][(k + flip) % 2]
            net[fc][r][c] = colors[k]

    return mx.array(net, dtype=mx.int32)


def print_net(net: mx.array) -> str:
    """Render the unfolded cross layout as text and print it."""
    n = net.tolist()

    def row(face, r):
        return ' '.join(ITOA[n[face][r][c]] for c in range(3))

    lines = []
    for r in range(3):
        lines.append('        ' + row(U, r))
    lines.append('')
    for r in range(3):
        lines.append('  '.join(row(fc, r) for fc in (L, F, R, B)))
    lines.append('')
    for r in range(3):
        lines.append('        ' + row(D, r))
    out = '\n'.join(lines)
    print(out)
    return out


if __name__ == "__main__":
    from state import MOVES

    print("solved:")
    print_net(state_to_net(State()))
    for name, move in MOVES.items():
        print(f"\n{name}:")
        print_net(state_to_net(move))
