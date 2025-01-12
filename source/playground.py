import torch
import typing
import math
import PIL

# mps test
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(device)

def inverse_permutation(p:torch.Tensor)->torch.Tensor:
    ip = torch.zeros_like(p, dtype=p.dtype)
    p = p.clone().to(torch.int64)
    for i in range(p.size(0)):
        ip[p[i]] = i
    return ip

def inverse_orientation(o:torch.Tensor)->torch.Tensor:
    o = o.clone()
    o[:,1] = -o[:,1]
    return o

State = typing.NamedTuple(
    "State", [
        ("corner_positions", torch.Tensor), 
        ("corner_orientations", torch.Tensor), 
        ("edge_positions", torch.Tensor), 
        ("edge_orientations", torch.Tensor)
    ]
)

class State:
    """_summary_
    
    # State
    Cube state representation.
    This is also used for the cube state transformation.
    The cube state is represented by 8 corner positions, 3 orientations for each corner,
    12 edge positions, and 2 orientations for each edge.
    The corner and edge positions are represented by the index of the corner or edge.
    The corner orientations are represented by the cosine and sine of the angle of the orientation.
    The edge orientations are represented by the cosine and sine of the angle of the orientation.
    """
    def __init__(
            self,
            corner_positions:torch.Tensor=None,
            corner_orientations:torch.Tensor=None,
            edge_positions:torch.Tensor=None,
            edge_orientations:torch.Tensor=None):
        """_summary_
        
        Args:
            corner_positions (torch.Tensor, optional): [8 corners]. Defaults to None.
            corner_orientations (torch.Tensor, optional): [2dims for 3 orientations [cos, sin] * 8]. Defaults to None.
            edge_positions (torch.Tensor, optional): [12 edges]. Defaults to None.
            edge_orientations (torch.Tensor, optional): [2dims for 2 orientations [cos, sin] * 12]. Defaults to None.
        """
        
        none_all = (
            corner_positions is None
            and corner_orientations is None 
            and edge_positions is None 
            and edge_orientations is None)
        not_none_all = (
            corner_positions is not None 
            and corner_orientations is not None 
            and edge_positions is not None 
            and edge_orientations is not None)
        
        if none_all:
            self.corner_positions = torch.tensor(
                list(range(8)), dtype=torch.uint8)
            self.corner_orientations = torch.tensor(
                [[1,0]] * 8, dtype=torch.float16)
            self.edge_positions = torch.tensor(
                list(range(12)), dtype=torch.uint8)
            self.edge_orientations = torch.tensor(
                [[1,0]] * 12, dtype=torch.float16)
        elif not_none_all:
            self.corner_positions = corner_positions
            self.corner_orientations = corner_orientations
            self.edge_positions = edge_positions
            self.edge_orientations = edge_orientations
        else:
            raise ValueError(
                "Either all or none of the state components must be provided.")
            
    def clone(self):
        return State(
            corner_positions=self.corner_positions.clone(),
            corner_orientations=self.corner_orientations.clone(),
            edge_positions=self.edge_positions.clone(),
            edge_orientations=self.edge_orientations.clone())
        
    def _half_apply(self,
                    p:torch.Tensor, o:torch.Tensor,
                    sp:torch.Tensor, so:torch.Tensor)->tuple[torch.Tensor]:
        """_summary_
        immutable
        """
        
        p = p[sp.to(torch.int64)].clone()
        so = so[sp.to(torch.int64)].clone()
        o = torch.stack([
            # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            o[:,0] * so[:,0] - o[:,1] * so[:,1],
            # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
            o[:,1] * so[:,0] + o[:,0] * so[:,1],
        ], dim=1).clone()
        return (p, o)
    
    def _apply(self, state:State)->tuple[torch.Tensor]:
        cp, co = self._half_apply(
            self.corner_positions, self.corner_orientations,
            state.corner_positions, state.corner_orientations)
        ep, eo = self._half_apply(
            self.edge_positions, self.edge_orientations,
            state.edge_positions, state.edge_orientations)
        return (cp, co, ep, eo)
    
    def apply(self, state:State):
        """_summary_

        mutate self state by applying the given state.

        Args:
            state (State): to be applied state
        """
        cp, co, ep, eo = self._apply(state)
        self.corner_positions = cp
        self.corner_orientations = co
        self.edge_positions = ep
        self.edge_orientations = eo
        
    def get_applied(self, state:State)->State:
        """_summary_

        get the new state by applying the given state.
        immutable.
        
        Args:
            state (State): to be applied state

        Returns:
            State: new state after applying the given state
        """
        
        cp, co, ep, eo = self._apply(state)
        return State(
            corner_positions=cp,
            corner_orientations=co,
            edge_positions=ep,
            edge_orientations=eo)
        
    def __str__(self):
        return (
            f"corner_positions {self.corner_positions.size()}: {self.corner_positions}\n"
            f"corner_orientations {self.corner_orientations.size()}: {self.corner_orientations}\n"
            f"edge_positions {self.edge_positions.size()}: {self.edge_positions}\n"
            f"edge_orientations {self.edge_orientations.size()}: {self.edge_orientations}\n"
        )
        
    def __add__(self, state:State)->State:
        return self.get_applied(state)
    
    def __iadd__(self, state:State)->State:
        self.apply(state)
        return self
    
    def __sub__(self, state:State)->State:
        return self.get_applied(-state)
    
    def __isub__(self, state:State)->State:
        self.apply(-state)
        return self
    
    def __neg__(self)->State:
        icp = inverse_permutation(self.corner_positions)
        iep = inverse_permutation(self.edge_positions)
        
        ico = inverse_orientation(self.corner_orientations)[icp.to(torch.int64)]
        ieo = inverse_orientation(self.edge_orientations)[iep.to(torch.int64)]
        
        return State(
            corner_positions=icp,
            corner_orientations=ico,
            edge_positions=iep,
            edge_orientations=ieo)
        
    def __mul__(self, n:int)->State:
        assert isinstance(n, int), "Not Defined multiplication by non-integer"
        assert n != 0, "Not Defined multiplication by 0"
        
        s = self.clone()
        move = self.clone()
        if n < 0:
            move = -move
        for _ in range(abs(n)):
            s = s + move
        return s

    def __rmul__(self, n:int)->State:
        return self.__mul__(n)
    
    def __eq__(self, state:State)->bool:
        cp = torch.equal(self.corner_positions, state.corner_positions)
        co = torch.allclose(self.corner_orientations, state.corner_orientations)
        ep = torch.equal(self.edge_positions, state.edge_positions)
        eo = torch.allclose(self.edge_orientations, state.edge_orientations)
        return cp and co and ep and eo


# test_tensor = torch.tensor(list(range(8)), dtype=torch.uint8)
# slice_tensor = torch.tensor([0, 3, 5, 7], dtype=torch.uint8)
# print(test_tensor)
# print(test_tensor[slice_tensor.to(torch.int64)])
# exit()


# r
# [0, 2, 6, 3, 4, 1, 5, 7],
# [0, 1, 2, 0, 0, 2, 1, 0],
# [0, 5, 9, 3, 4, 2, 6, 7, 8, 1, 10, 11],
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 回転 120度 2/3π
pi = math.pi
rot0 = [math.cos(0), math.sin(0)]
rot1 = [math.cos(2/3*pi), math.sin(2/3*pi)]
rot2 = [math.cos(4/3*pi), math.sin(4/3*pi)]

r = State(
    corner_positions=torch.tensor(
        [0, 2, 6, 3, 4, 1, 5, 7], dtype=torch.uint8),
    corner_orientations=torch.tensor(
        [rot0, rot1, rot2, rot0, rot0, rot2, rot1, rot0], dtype=torch.float16),
    edge_positions=torch.tensor(
        [0, 5, 9, 3, 4, 2, 6, 7, 8, 1, 10, 11], dtype=torch.uint8),
    edge_orientations=torch.tensor(
        [rot0] * 12, dtype=torch.float16),
)

s = State()
# print(s)
r = s.get_applied(r)
r2 = r * 2
r3 = r * 3
r4 = r3 + r

print('e', s.corner_positions)
print('e r', r.corner_positions)
print('e r r', r2.corner_positions)
print('e r r r', r3.corner_positions)
print('e r r r r', r4.corner_positions)

print("r", r)
print("r'", -r)
print("r' r", -r + r)

print("r2", -2*r)
print("r2", 2*r)
print("r2 == -r2", 2*r == -2*r)

# print((r3+r3).corner_positions)


# 表示してみる
# 画像のtensorを作ってPILで表示
# 画像のサイズ
size = 512
# 画像の背景色

black = (0, 0, 0)
bg_color = black

# h w c
image = torch.ones((size, size, 3), dtype=torch.uint8)
image = image * torch.tensor(bg_color, dtype=torch.uint8)

"""
  Y
O G R B
  W
で並べる
"""

# 1つのキューブのサイズ
cube_size = size // 4
sub_cube_size = cube_size // 3

# 展開図は英語でnet
# 

def state_to_net(state:State)->torch.Tensor:
    # f b l r u d
    # 0 1 2 3 4 5
    net = torch.zeros((6, 3, 3), dtype=torch.uint8)
    # 面ごとに向きに対応する数字を入れていく
    # 真ん中にはそれぞれ面の向き（インデックスと同じ）を入れる
    net[0, 2, 2] = 0
    net[1, 2, 2] = 1
    net[2, 2, 2] = 2
    net[3, 2, 2] = 3
    net[4, 2, 2] = 4
    net[5, 2, 2] = 5
    
    # それぞれの面に対して、角と辺の位置を入れていく

