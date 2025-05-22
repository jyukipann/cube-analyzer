from state import State, TWIST_TABLE
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
# 面の番号が少ない方から時計回りが面の番号が入る
# その場所にあるサブキューブの向きが入る
CORNER_FACES = torch.tensor([
    [L, U, B], # 0
    [R, U, B], # 1
    [F, U, R], # 2
    [F, L, U], # 3
    [L, B, D], # 4
    [R, D, B], # 5
    [F, R, D], # 6
    [F, D, L], # 7
], dtype=torch.int64)

# F00, F02, F20, F22のどれか
# その角を正面から見て面の番号が最小のものから時計まわりに入れていく
CORNER_SUBCUBES = torch.tensor([
    [FP00, FP00, FP02], # 0
    [FP02, FP02, FP00], # 1
    [FP02, FP22, FP00], # 2
    [FP00, FP02, FP20], # 3
    [FP20, FP22, FP20], # 4
    [FP22, FP22, FP20], # 5
    [FP22, FP20, FP02], # 6
    [FP20, FP00, FP22], # 7
], dtype=torch.int64)

EDGE_FACES = torch.tensor([
    [L,B], # E00
    [R,B], # E01
    [F,R], # E02
    [F,L], # E03
    [B,U], # E04
    [R,U], # E05
    [F,U], # E06
    [L,U], # E07
    [B,D], # E08
    [R,D], # E09
    [F,D], # E10
    [L,D], # E11
], dtype=torch.int64)

EDGE_SUBCUBES = torch.tensor([
    [FP10, FP12], # E00
    [FP12, FP10], # E01
    [FP12, FP10], # E02
    [FP10, FP12], # E03
    [FP01, FP01], # E04
    [FP01, FP12], # E05
    [FP01, FP21], # E06
    [FP01, FP10], # E07
    [FP21, FP21], # E08
    [FP21, FP12], # E09
    [FP21, FP01], # E10
    [FP21, FP10], # E11
], dtype=torch.int64)


def state_to_net(state: State)->torch.Tensor:
    net = torch.zeros((6, 3, 3), dtype=torch.int8)+9
    
    # center
    (
        net[F00, *FP11], net[F01, *FP11], net[F02, *FP11],
        net[F03, *FP11], net[F04, *FP11], net[F05, *FP11],
    ) = tuple(range(6))
    
    # conner
    slicer_cp = state.corner_positions.to(torch.int64)
    corner_faces = CORNER_FACES[slicer_cp]
    corner_twists = state.twist_co
    corner_subcube_positions = CORNER_SUBCUBES
    ## corner_facesをtwist分だけ回転させる
    corner_twists = corner_twists.unsqueeze(1).view(-1, 1).to(torch.int64) 
    corner_faces_rotated = torch.empty_like(corner_faces)
    for i in range(8):
        # subcube回転
        corner_faces_rotated[i] = torch.roll(corner_faces[i], -corner_twists[i].item())
        # subcubeの位置に値を入れる
        net[CORNER_FACES[i, 0], *corner_subcube_positions[i, 0]] = corner_faces_rotated[i][0]
        net[CORNER_FACES[i, 1], *corner_subcube_positions[i, 1]] = corner_faces_rotated[i][1]
        net[CORNER_FACES[i, 2], *corner_subcube_positions[i, 2]] = corner_faces_rotated[i][2]

    # edge
    slicer_ep = state.edge_positions.to(torch.int64)
    edge_faces = EDGE_FACES[slicer_ep]
    edge_twists = state.twist_eo
    edge_subcube_positions = EDGE_SUBCUBES
    for i in range(12):
        # subcubeフリップ
        is_flipped = edge_twists[i]
        # subcubeの位置に値を入れる
        net[EDGE_FACES[i, 0], *edge_subcube_positions[i, 0]] = edge_faces[i][is_flipped]
        net[EDGE_FACES[i, 1], *edge_subcube_positions[i, 1]] = edge_faces[i][(is_flipped+1)%2]
    
    print_net(net)

    return net

def print_net(net:torch.Tensor):
    net_for_print = f"""
        0-------1
        | {net[F04, *FP00]} {net[F04, *FP01]} {net[F04, *FP02]} |
        | {net[F04, *FP10]} {net[F04, *FP11]} {net[F04, *FP12]} |
        | {net[F04, *FP20]} {net[F04, *FP21]} {net[F04, *FP22]} |
0-------3-------2-------1-------0
| {net[F02, *FP00]} {net[F02, *FP01]} {net[F02, *FP02]} | {net[F00, *FP00]} {net[F00, *FP01]} {net[F00, *FP02]} | {net[F01, *FP00]} {net[F01, *FP01]} {net[F01, *FP02]} | {net[F03, *FP00]} {net[F03, *FP01]} {net[F03, *FP02]} |
| {net[F02, *FP10]} {net[F02, *FP11]} {net[F02, *FP12]} | {net[F00, *FP10]} {net[F00, *FP11]} {net[F00, *FP12]} | {net[F01, *FP10]} {net[F01, *FP11]} {net[F01, *FP12]} | {net[F03, *FP10]} {net[F03, *FP11]} {net[F03, *FP12]} |
| {net[F02, *FP20]} {net[F02, *FP21]} {net[F02, *FP22]} | {net[F00, *FP20]} {net[F00, *FP21]} {net[F00, *FP22]} | {net[F01, *FP20]} {net[F01, *FP21]} {net[F01, *FP22]} | {net[F03, *FP20]} {net[F03, *FP21]} {net[F03, *FP22]} |
4-------7-------6-------5-------4
        | {net[F05, *FP00]} {net[F05, *FP01]} {net[F05, *FP02]} |
        | {net[F05, *FP10]} {net[F05, *FP11]} {net[F05, *FP12]} |
        | {net[F05, *FP20]} {net[F05, *FP21]} {net[F05, *FP22]} |
        4-------5
    """
    print(net_for_print.replace("9", " "))

if __name__ == "__main__":
    TT = TWIST_TABLE.tolist()
    r = State(
        corner_positions=torch.tensor(
            [0, 2, 6, 3, 4, 1, 5, 7], dtype=torch.int8),
        corner_orientations=torch.tensor(
            [TT[0], TT[1], TT[2], TT[0], TT[0], TT[2], TT[1], TT[0]], dtype=torch.float16),
        edge_positions=torch.tensor(
            [0, 5, 9, 3, 4, 2, 6, 7, 8, 1, 10, 11], dtype=torch.int8),
        edge_orientations=torch.tensor(
            [1] * 12, dtype=torch.int8),
    )
    
    state_to_net(State())
    state_to_net(r)
    