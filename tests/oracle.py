"""Independent facelet ground-truth for the cube net.

This model knows nothing about the group-theoretic ``State``; it tracks the 54
stickers directly in 3D and rotates layers. It is the reference that
``state_to_net`` is validated against, so a shared bug cannot hide.

Coordinate system:
    x: L(-1) .. R(+1)   y: D(-1) .. U(+1)   z: B(-1) .. F(+1)
A sticker = (point, normal); a face turn rotates point and normal of every
sticker in the moving layer (clockwise viewed from outside that face).
"""

F, R, L, B, U, D = 0, 1, 2, 3, 4, 5
FACES = [F, R, L, B, U, D]
NAME_TO_FACE = {'F': F, 'R': R, 'L': L, 'B': B, 'U': U, 'D': D}

FACE_NORMAL = {
    F: (0, 0, 1), B: (0, 0, -1),
    R: (1, 0, 0), L: (-1, 0, 0),
    U: (0, 1, 0), D: (0, -1, 0),
}
NORMAL_FACE = {v: k for k, v in FACE_NORMAL.items()}


def _cell_point_normal(face, row, col):
    if face == U:
        p = (col - 1, 1, row - 1)
    elif face == F:
        p = (col - 1, 1 - row, 1)
    elif face == R:
        p = (1, 1 - row, 1 - col)
    elif face == L:
        p = (-1, 1 - row, col - 1)
    elif face == B:
        p = (1 - col, 1 - row, -1)
    else:  # D
        p = (col - 1, -1, 1 - row)
    return p, FACE_NORMAL[face]


def _rot(face, v):
    x, y, z = v
    if face == R:
        return (x, z, -y)
    if face == L:
        return (x, -z, y)
    if face == U:
        return (-z, y, x)
    if face == D:
        return (z, y, -x)
    if face == F:
        return (y, -x, z)
    return (-y, x, z)  # B


def _inv_rot(face, v):
    return _rot(face, _rot(face, _rot(face, v)))


def _in_layer(face, p):
    x, y, z = p
    return {R: x == 1, L: x == -1, U: y == 1,
            D: y == -1, F: z == 1, B: z == -1}[face]


def oracle_net(seq):
    """Net (list[6][3][3] of face indices) after applying ``seq`` to solved.

    ``seq`` is a list of face indices or single-letter move names.
    """
    seq = [NAME_TO_FACE[m] if isinstance(m, str) else m for m in seq]
    net = [[[None] * 3 for _ in range(3)] for _ in range(6)]
    for face in FACES:
        for row in range(3):
            for col in range(3):
                p, n = _cell_point_normal(face, row, col)
                for mv in reversed(seq):
                    if _in_layer(mv, p):
                        p = _inv_rot(mv, p)
                        n = _inv_rot(mv, n)
                net[face][row][col] = NORMAL_FACE[n]
    return net
