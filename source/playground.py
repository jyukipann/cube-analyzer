import torch
from math import pi
import math
from cube import TWIST_TABLE, State

# r
# [0, 2, 6, 3, 4, 1, 5, 7],
# [0, 1, 2, 0, 0, 2, 1, 0],
# [0, 5, 9, 3, 4, 2, 6, 7, 8, 1, 10, 11],
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

print(r)
print(r.twist_eo, r.twist_co)

s = State()
# print(s)
r_state = s.get_applied(r)

print(r == r_state)

print(r)
print(r_state)

r2 = r * 2
r3 = r * 3
r4 = r3 + r

print(r2)
print(r3)
print(r4)

# print('e', s.corner_positions)
# print('e r', r.corner_positions)
# print('e r r', r2.corner_positions)
# print('e r r r', r3.corner_positions)
# print('e r r r r', r4.corner_positions)


# print("r'", -r)
# print("r' r", -r + r)

# print("r2", -2*r)
# print("r2", 2*r)
# print("r2 == -r2", 2*r == -2*r)

# print((r3+r3).corner_positions)


# 表示してみる
# 画像のtensorを作ってPILで表示
# 画像のサイズ
size = 512
# 画像の背景色

black = (0, 0, 0)
bg_color = black

# h w c
image = torch.ones((size, size, 3), dtype=torch.int8)
image = image * torch.tensor(bg_color, dtype=torch.int8)

# 1つのキューブのサイズ
cube_size = size // 4
sub_cube_size = cube_size // 3

# 展開図は英語でnet

def state_to_net(state:State)->torch.Tensor:
    # f b l r u d
    # 0 1 2 3 4 5
    net = torch.zeros((6, 3, 3), dtype=torch.int8)
    # 面ごとに向きに対応する数字を入れていく
    # 真ん中にはそれぞれ面の向き（インデックスと同じ）を入れる
    net[F00, FP11] = 0
    net[F01, FP11] = 1
    net[F02, FP11] = 2
    net[F03, FP11] = 3
    net[F04, FP11] = 4
    net[F05, FP11] = 5
    
    # それぞれの面に対して、角と辺の位置を入れていく
    # 角と面の向きの関係
    # 面の番号が少ない方から時計回りが面の番号が入る
    # その場所にあるサブキューブの向きが入る
    coner_faces = torch.tensor([
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
    corner_subcube_positons = torch.tensor([
        [FP00, FP02, FP00], # 0
        [FP02, FP02, FP00], # 1
        [FP02, FP22, FP00], # 2
        [FP00, FP02, FP20], # 3
        [FP02, FP02, FP20], # 4
        [FP22, FP22, FP20], # 5
        [FP22, FP20, FP02], # 6
        [FP20, FP22, FP00], # 7
    ], dtype=torch.int64)
    
    edge_faces = torch.tensor([
        [F01, F02], [F01, F03], [F00, F03], [F00, F02],
        [F01, F04], [F03, F04], [F00, F04], [F02, F04],
        [F01, F05], [F03, F05], [F00, F05], [F02, F05],
    ], dtype=torch.int64)
    
    # 角の色を入れていく
    for target_cp in range(8):
        # 角の位置
        cp = state.corner_positions[target_cp].item()
        # cpはiの位置にどの角があるかの番号を示す
        cp_faces = coner_faces[cp]
        # cp_facesはcpの位置にある角の面の番号（色）が入っている
        
        # 角の向き
        co = state.corner_orientations[target_cp]
        # 角の向きから面の向きを求める
        # 角の向きはcos, sinなので、arctanで求める
        angle = math.atan2(co[1].item(), co[0].item())
        # 角の向きから面の向きを求める
        face = int(angle / (2/3*pi))
        # 角の位置から面の位置を求める
        
        # print(cp_faces)
        net[
            coner_faces[target_cp, 0], 
            *corner_subcube_positons[target_cp, 0]
        ] = cp_faces[(0+face)%3]
        net[
            coner_faces[target_cp, 1], 
            *corner_subcube_positons[target_cp, 1]
        ] = cp_faces[(1+face)%3]
        net[
            coner_faces[target_cp, 2], 
            *corner_subcube_positons[target_cp, 2]
        ] = cp_faces[(2+face)%3]
        
    for target_ep in range(12):
        ep = state.edge_positions[target_ep].item()
        
        ep_faces = edge_faces[ep]
        
        eo = state.edge_orientations[target_ep]
        
        angle = (eo - 1)/-2
        
        
        
    return net

# r_net = state_to_net(r)
# s = State()
# print_net(state_to_net(State()))
# print_net(r_net)
# print()
# # print_net(state_to_net(r4))