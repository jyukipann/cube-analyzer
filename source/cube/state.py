import torch
import typing
import math
from math import pi

# mps test
# device = torch.device("mps" if torch.mps.is_available() else "cpu")
# print(device)

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

def inverse_orientation_int(o:torch.Tensor)->torch.Tensor:
    return -o.clone()

TWIST_TABLE = torch.tensor([
    [math.cos(0), math.sin(0)],   # 0
    [math.cos(2/3*pi), math.sin(2/3*pi)],   # 1
    [math.cos(4/3*pi), math.sin(4/3*pi)],   # 2
], dtype=torch.float16)

def get_twist_vec(twist:int, to_list=True)->torch.Tensor:
    """_summary_

    Args:
        twist (int): twist index

    Returns:
        torch.Tensor: twist vector
    """
    twist = twist % 3
    ret = TWIST_TABLE[twist].clone()
    if to_list:
        ret = ret.tolist()
    return ret

State = typing.NamedTuple(
    "State", [
        ("corner_positions", torch.Tensor|list[int]), 
        ("corner_orientations", torch.Tensor|list[list[float]]|list[int]), 
        ("edge_positions", torch.Tensor|list[int]), 
        ("edge_orientations", torch.Tensor|list[int]),
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
            corner_positions:torch.Tensor|list[int]=None,
            corner_orientations:torch.Tensor|list[list[float]]|list[int]=None,
            edge_positions:torch.Tensor|list[int]=None,
            edge_orientations:torch.Tensor|list[list[float]]|list[int]=None):
        """_summary_
        
        Args:
            corner_positions (torch.Tensor, optional): [8 corners]. Defaults to None.
            corner_orientations (torch.Tensor, optional): [2dims for 3 orientations [cos, sin] * 8]. Defaults to None.
            edge_positions (torch.Tensor, optional): [12 edges]. Defaults to None.
            edge_orientations (torch.Tensor, optional): [(+/-)1 for 2 orientations -1/+1 * 12]. Defaults to None.
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
                list(range(8)), dtype=torch.int8)
            self.corner_orientations = torch.tensor(
                [[1,0]] * 8, dtype=torch.float16)
            self.edge_positions = torch.tensor(
                list(range(12)), dtype=torch.int8)
            self.edge_orientations = torch.tensor(
                [1] * 12, dtype=torch.int8)
        elif not_none_all:
            if isinstance(corner_positions, list):
                corner_positions = torch.tensor(
                    corner_positions, dtype=torch.int8)
            if isinstance(corner_orientations, list):
                _corner_orientations = torch.tensor(corner_orientations)
                if _corner_orientations.size() == (8, 2):
                    _corner_orientations = _corner_orientations.to(torch.float16)
                elif _corner_orientations.size() == (8,):
                    twist_vecs = [torch.tensor(get_twist_vec(t)) for t in corner_orientations]
                    _corner_orientations = torch.stack(twist_vecs, dim=0).to(torch.float16)
                else:
                    raise ValueError(
                        "corner_orientations must be of size (8, 2) or (8,)")
                corner_orientations = _corner_orientations
            if isinstance(edge_positions, list):
                edge_positions = torch.tensor(
                    edge_positions, dtype=torch.int8)
            if isinstance(edge_orientations, list):
                edge_orientations = torch.tensor(
                    edge_orientations, dtype=torch.int8)
            self.corner_positions = corner_positions
            self.corner_orientations = corner_orientations
            self.edge_positions = edge_positions
            self.edge_orientations = edge_orientations
        else:
            raise ValueError(
                "Either all or none of the state components must be provided.")
    
    def to(self, t)->State:
        return State(
            corner_positions=self.corner_positions.to(t),
            corner_orientations=self.corner_orientations.to(t),
            edge_positions=self.edge_positions.to(t),
            edge_orientations=self.edge_orientations.to(t))
            
    def clone(self)->State:
        return State(
            corner_positions=self.corner_positions.clone(),
            corner_orientations=self.corner_orientations.clone(),
            edge_positions=self.edge_positions.clone(),
            edge_orientations=self.edge_orientations.clone())
        
    def _corner_apply(self,
                    p:torch.Tensor, o:torch.Tensor,
                    sp:torch.Tensor, so:torch.Tensor)->tuple[torch.Tensor]:
        """_summary_
        immutable
        """
        
        p = p[sp.to(torch.int64)].clone()
        so = so[sp.to(torch.int64)].clone()
        o = torch.stack([
            # cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            o[:,0] * so[:,0] + o[:,1] * so[:,1],
            # sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
            o[:,1] * so[:,0] - o[:,0] * so[:,1],
        ], dim=1)
        return (p, o)
    
    def _edge_apply(self,
                    p:torch.Tensor, o:torch.Tensor,
                    sp:torch.Tensor, so:torch.Tensor)->tuple[torch.Tensor]:
        """_summary_
        immutable
        """
        
        p = p[sp.to(torch.int64)].clone()
        so = so[sp.to(torch.int64)].clone()
        o = o*so
        return (p, o)
    
    
    def _apply(self, state:State)->tuple[torch.Tensor]:
        cp, co = self._corner_apply(
            self.corner_positions, self.corner_orientations,
            state.corner_positions, state.corner_orientations)
        ep, eo = self._edge_apply(
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
            f"corner_positions {self.corner_positions.size()}: {self.corner_positions.tolist()}\n"
            f"corner_orientations {self.corner_orientations.size()}: {self.twist_co.tolist()}\n"
            f"edge_positions {self.edge_positions.size()}: {self.edge_positions.tolist()}\n"
            f"edge_orientations {self.edge_orientations.size()}: {self.twist_eo.tolist()}\n"
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
        ieo = inverse_orientation_int(self.edge_orientations)[iep.to(torch.int64)]
        
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
        eo = torch.equal(self.edge_orientations, state.edge_orientations)
        return cp and co and ep and eo
    
    @property
    def twist_eo(self)->torch.Tensor:
        return ((self.edge_orientations.clone() - 1)/-2).to(torch.int8)
    
    @property
    def twist_co(self)->torch.Tensor:
        co = (
            self.corner_orientations 
            / 
            torch.linalg.norm(self.corner_orientations)
        )

        idx = torch.argmax(co @ TWIST_TABLE.T, dim=1).to(torch.int8)
        return idx
