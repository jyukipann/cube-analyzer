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

# 面内でのサブキューブの位置
FP00, FP01, FP02 =((0,0), (0,1), (0,2))
FP10, FP11, FP12 =((1,0), (1,1), (1,2))
FP20, FP21, FP22 =((2,0), (2,1), (2,2))

# それぞれの面に対して、角と辺の位置を入れていく
# 角と面の向きの関係
# 面の番号が少ない方から時計回りが面の番号が入る
# その場所にあるサブキューブの向きが入る
CORNER_FACES = torch.tensor([
    [F02, F03, F04], # 0 LUB
    [F01, F04, F03], # 1 RUB
    [F00, F04, F01], # 2 FUR
    [F00, F02, F04], # 3 FLU
    [F02, F05, F03], # 4 LBD
    [F01, F03, F05], # 5 RDB
    [F00, F01, F05], # 6 FRD
    [F00, F05, F02], # 7 FDL
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

def state_to_net(state: State)->torch.Tensor:
    net = torch.zeros((6, 3, 3), dtype=torch.int8)
    
    (
        net[F00, *FP11], net[F01, *FP11], net[F02, *FP11],
        net[F03, *FP11], net[F04, *FP11], net[F05, *FP11],
    ) = tuple(range(6))
    
    slicer_cp = state.corner_positions.to(torch.int64)
    corner_faces = CORNER_FACES[slicer_cp]
    corner_twists = state.twist_co
    corner_subcube_positions = CORNER_SUBCUBES[slicer_cp]
    print(corner_twists)
    print(corner_faces)
    # print(corner_subcube_positions)
    
    # corner_facesをtwist分だけ回転させる
    corner_twists = corner_twists.unsqueeze(1).view(-1, 1).to(torch.int64)
    corner_faces_twisted = torch.empty_like(corner_faces)
    print(corner_faces_twisted[:, 0])
    print(corner_faces[:, (0+corner_twists[:, 0]) % 3])
    print(corner_twists)
    corner_faces_twisted[:, 0] = corner_faces[:, (0+corner_twists[:, 0]) % 3]
    corner_faces_twisted[:, 1] = corner_faces[:, (1+corner_twists[:, 0]) % 3]
    corner_faces_twisted[:, 2] = corner_faces[:, (2+corner_twists[:, 0]) % 3]
    print(corner_faces_twisted)
    return net
    
    

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
    