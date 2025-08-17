from state import State
import torch

"""
  W
R G O B
  Y

  U
L F R B
  D
  
  4
2 0 1 3
  5
で並べる

X: R L
Y: U D
Z: F B

各面は展開図において上から見て以下のようなナンバリングになる
[[0,0], [0,1], [0,2],]
[[1,0], [1,1], [1,2],]
[[2,0], [2,1], [2,2],]

              0-------------1
              | C00 E04 C01 |
              | E07 F04 E05 |
              | C03 E06 C02 |
0-------------3-------------2-------------1-------------0
| C00 E07 C03 | C03 E06 C02 | C02 E05 C01 | C01 E04 C00 |
| E00 F02 E03 | E03 F00 E02 | E02 F01 E01 | E01 F03 E00 |
| C04 E11 C07 | C07 E10 C06 | C06 E09 C05 | C05 E08 C04 |
4-------------7-------------6-------------5-------------4
              | C07 E10 C06 |
              | E11 F05 E09 |
              | C04 E08 C05 |
              4-------------5
"""
# 角の位置
C00, C01, C02, C03, C04, C05, C06, C07 = tuple(range(8))
# 辺の位置
E00, E01, E02, E03, E04, E05, E06, E07, E08, E09, E10, E11 = tuple(range(12))
# 面の位置
F00, F01, F02, F03, F04, F05 = tuple(range(6))
F, R, L, B, U, D = F00, F01, F02, F03, F04, F05
ITOA = {0:'F', 1:'R', 2:'L', 3:'B', 4:'U', 5:'D'}

# 面内でのサブキューブの位置
FP00, FP01, FP02 = ((0,0), (0,1), (0,2))
FP10, FP11, FP12 = ((1,0), (1,1), (1,2))
FP20, FP21, FP22 = ((2,0), (2,1), (2,2))

# それぞれの面に対して、角と辺の位置を入れていく
# 角と面の向きの関係
# その場所にあるサブキューブの向きが入る
# Xから始まり、反時計回りに入れていく
CORNER_FACES = torch.tensor([
    [L, B, U,], # 0
    [R, U, B,], # 1
    [R, F, U,], # 2
    [L, U, F,], # 3
    [L, D, B,], # 4
    [R, D, B,], # 5
    [R, D, F,], # 6
    [L, D, F,], # 7
], dtype=torch.int64)

# F00, F02, F20, F22のどれか
CORNER_SUBCUBES = torch.tensor([
    [FP00, FP02, FP00,], # 0
    [FP02, FP02, FP00,], # 1
    [FP00, FP02, FP22,], # 2
    [FP02, FP20, FP00,], # 3
    [FP20, FP20, FP22,], # 4
    [FP22, FP22, FP20,], # 5
    [FP20, FP02, FP22,], # 6
    [FP22, FP00, FP20,], # 7
], dtype=torch.int64)

EDGE_FACES = torch.tensor([
    [L, B,], # E00
    [R, B,], # E01
    [R, F,], # E02
    [L, F,], # E03
    [U, B,], # E04
    [R, U,], # E05
    [U, F,], # E06
    [L, U,], # E07
    [D, B,], # E08
    [R, D,], # E09
    [D, F,], # E10
    [L, D,], # E11
], dtype=torch.int64)

EDGE_SUBCUBES = torch.tensor([
    [FP10, FP12,], # E00
    [FP12, FP10,], # E01
    [FP10, FP12,], # E02
    [FP12, FP10,], # E03
    [FP01, FP01,], # E04
    [FP01, FP12,], # E05
    [FP21, FP01,], # E06
    [FP01, FP10,], # E07
    [FP21, FP21,], # E08
    [FP21, FP12,], # E09
    [FP01, FP21,], # E10
    [FP21, FP10,], # E11
], dtype=torch.int64)


def state_to_net(state: State)->torch.Tensor:
    net = torch.ones((6, 3, 3), dtype=torch.int8)
    for c in range(6):
        net[c] *= c
    
    _state_to_net_corner(state, net)
    _state_to_net_edge(state, net)

    return net

# twistとsubecubeの展開の対応関係
TWIST_SUBCUBE_MAP = {
    C00: [(2, 0, 1), (0, 1, 2), (0, 1, 2)],
    C01: [(1, 2, 0), (0, 1, 2), (0, 2, 1)],
    C02: [(2, 0, 1), (0, 1, 2), (0, 1, 2)],
    C03: [(1, 2, 0), (0, 1, 2), (0, 2, 1)],
    C04: [(2, 1, 0), (0, 2, 1), (0, 1, 2)],
    C05: [(2, 1, 0), (0, 2, 1), (0, 2, 1)],
    C06: [(2, 1, 0), (0, 2, 1), (0, 1, 2)],
    C07: [(2, 1, 0), (0, 2, 1), (0, 2, 1)],
}

def _state_to_net_corner(state: State, net: torch.Tensor) -> None:

    twists = state.twist_co  # ← ピースID基準のねじれ(0,1,2)

    for position in range(8):  # 位置ID
        corner_id = int(state.corner_positions[position])
        twist = int(twists[corner_id])

        destination_colors = CORNER_FACES[position]
        source_colors = CORNER_FACES[corner_id]

        destination_position = CORNER_SUBCUBES[position]
        source_position = CORNER_SUBCUBES[corner_id]

        net[destination_colors[0], *destination_position[0]] = source_colors[(0+twist)%3]
        net[destination_colors[1], *destination_position[1]] = source_colors[(1+twist)%3]
        net[destination_colors[2], *destination_position[2]] = source_colors[(2+twist)%3]


FLIP_SUBCUBE_MAP = {
    E00: [(0, 1), (0, 1)],
    E01: [(0, 1), (0, 1)],
    E02: [(0, 1), (0, 1)],
    E03: [(0, 1), (0, 1)],
    E04: [(0, 1), (0, 1)],
    E05: [(0, 1), (0, 1)],
    E06: [(0, 1), (0, 1)],
    E07: [(0, 1), (0, 1)],
    E08: [(0, 1), (0, 1)],
    E09: [(0, 1), (0, 1)],
    E10: [(0, 1), (0, 1)],
    E11: [(0, 1), (0, 1)],   
}

def _state_to_net_edge(state: State, net: torch.Tensor) -> None:
    subcube_map = FLIP_SUBCUBE_MAP

    flips = state.twist_eo  # ← ピースID基準 (0/1)

    for position in range(12):
        edge_id = int(state.edge_positions[position])

        flip = int(flips[edge_id])
        destination_colors = EDGE_FACES[position]
        source_colors = EDGE_FACES[edge_id]
        destination_position = EDGE_SUBCUBES[position]
        source_position = EDGE_SUBCUBES[edge_id]
        
        net[destination_colors[0], *destination_position[0]] = source_colors[0+flip]
        net[destination_colors[1], *destination_position[1]] = source_colors[1-flip]



def print_net(net:torch.Tensor):
    net_for_print = f"""
        0---★---1
        | {net[F04, *FP00]} {net[F04, *FP01]} {net[F04, *FP02]} |
        ○ {net[F04, *FP10]} {net[F04, *FP11]} {net[F04, *FP12]} ●
        | {net[F04, *FP20]} {net[F04, *FP21]} {net[F04, *FP22]} |
0---○---3-------2---●---1---★---0
| {net[F02, *FP00]} {net[F02, *FP01]} {net[F02, *FP02]} | {net[F00, *FP00]} {net[F00, *FP01]} {net[F00, *FP02]} | {net[F01, *FP00]} {net[F01, *FP01]} {net[F01, *FP02]} | {net[F03, *FP00]} {net[F03, *FP01]} {net[F03, *FP02]} |
■ {net[F02, *FP10]} {net[F02, *FP11]} {net[F02, *FP12]} | {net[F00, *FP10]} {net[F00, *FP11]} {net[F00, *FP12]} | {net[F01, *FP10]} {net[F01, *FP11]} {net[F01, *FP12]} | {net[F03, *FP10]} {net[F03, *FP11]} {net[F03, *FP12]} ■
| {net[F02, *FP20]} {net[F02, *FP21]} {net[F02, *FP22]} | {net[F00, *FP20]} {net[F00, *FP21]} {net[F00, *FP22]} | {net[F01, *FP20]} {net[F01, *FP21]} {net[F01, *FP22]} | {net[F03, *FP20]} {net[F03, *FP21]} {net[F03, *FP22]} |
4---●---7-------6---○---5---□---4
        | {net[F05, *FP00]} {net[F05, *FP01]} {net[F05, *FP02]} |
        ● {net[F05, *FP10]} {net[F05, *FP11]} {net[F05, *FP12]} ○
        | {net[F05, *FP20]} {net[F05, *FP21]} {net[F05, *FP22]} |
        4---□---5
    """
    print(net_for_print.replace("9", " "))

if __name__ == "__main__":
    from state import MOVES as moves
    
    old_map_corner = {
        C00: [(2, 0, 1), (0, 1, 2), (0, 1, 2)],
        C01: [(1, 2, 0), (0, 1, 2), (0, 2, 1)],
        C02: [(2, 0, 1), (0, 1, 2), (0, 1, 2)],
        C03: [(1, 2, 0), (0, 1, 2), (0, 2, 1)],
        C04: [(2, 1, 0), (0, 2, 1), (0, 1, 2)],
        C05: [(2, 1, 0), (0, 2, 1), (0, 2, 1)],
        C06: [(2, 1, 0), (0, 2, 1), (0, 1, 2)],
        C07: [(2, 1, 0), (0, 2, 1), (0, 2, 1)],
    }
    
    pn = print_net
    stn = state_to_net
    
    
    r = moves['R']
    l = moves['L']
    u = moves['U']
    d = moves['D']
    f = moves['F']
    b = moves['B']

    print("r @ l")
    print(r @ l)
    pn(stn(r @ l))
    print("u @ d")
    print(u @ d)
    pn(stn(u @ d))
    print("f @ b")
    print(f @ b)
    pn(stn(f @ b))

    print(torch.all(stn(r@l) == stn(r@l, old_map_corner)))
    print(torch.all(stn(~(r@l)) == stn(~(r@l), old_map_corner)))
    print(torch.all(stn(f@b) == stn(f@b, old_map_corner)))
    print(torch.all(stn(~(f@b)) == stn(~(f@b), old_map_corner)))
    print(torch.all(stn(u@d) == stn(u@d, old_map_corner)))
    print(torch.all(stn(~(u@d)) == stn(~(u@d), old_map_corner)))