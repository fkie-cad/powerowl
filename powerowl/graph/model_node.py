import dataclasses
import enum
import sys
import types
import warnings
from typing import TYPE_CHECKING, Optional, Any, Union, Tuple, List

from powerowl.graph.constants import FACILITY, OU
from powerowl.graph.enums import EdgeType
from powerowl.layers.powergrid.elements import GridElement

if TYPE_CHECKING:
    from powerowl.graph.graph_layer import GraphLayer
    from powerowl.graph import MultiLayerGraph
    from powerowl.layers.facilities import Facility
    from powerowl.layers.ou import OrganizationalUnit


@dataclasses.dataclass(eq=False, kw_only=True)
class ModelNode:
    name: Union[enum.Enum, str] = None
    layer: Optional['GraphLayer'] = None
    z_offset: float = 0
    color: Optional[str] = None

    def __post_init__(self):
        from powerowl.power_owl import PowerOwl
        self.id: int = PowerOwl.next_global_id()
        self._attributes = {}
        if self.name is None:
            self.name = f"{self.__class__.__name__}.{self.id}"
        elif isinstance(self.name, enum.Enum):
            self.name = f"{self.name.__class__.__name__}.{self.name.value}"

    def to_dict(self):
        fields = dataclasses.fields(self)
        data = {
            "id": self.id,
            "attributes": self.attributes
        }
        for field in fields:
            f_name = field.name
            data[f_name] = getattr(self, f_name)
        return data

    def from_dict(self, d: dict):
        self.id = d.pop("id")
        self._attributes = d.pop("attributes")
        field_names = [field.name for field in dataclasses.fields(self)]
        for f_name in d:
            if f_name not in field_names:
                raise AttributeError(f"Unknown field {f_name} for {self.__class__.__name__}")
            setattr(self, f_name, d[f_name])

    @property
    def uid(self) -> str:
        return str(self.id)

    @property
    def attributes(self) -> dict:
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = attributes

    def set(self, attribute: str, value: Any):
        self.attributes[attribute] = value

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __getitem__(self, item):
        return self.attributes.get(item)

    @property
    def mlg(self) -> 'MultiLayerGraph':
        return self.layer.mlg

    def get_facility(self, allow_cache: bool = True) -> Optional['Facility']:
        if allow_cache and self[FACILITY] is not None:
            return self[FACILITY]
        if not self.layer.mlg.has_node(self):
            return None
        facility = self.layer.mlg.owl.get_facility(self)
        self[FACILITY] = facility
        return facility

    def get_ou(self, allow_cache: bool = True) -> Optional['OrganizationalUnit']:
        if allow_cache and self[OU] is not None:
            return self[OU]
        if not self.layer.mlg.has_node(self):
            return None
        ou = self.layer.mlg.owl.get_ou(self)
        self[OU] = ou
        return ou

    def set_facility(self, facility: Optional[Union[str, 'Facility']]):
        old_facility = self.get_facility(False)
        if facility is not None:
            facility = self.mlg.get_model_node(facility)
        self[FACILITY] = facility
        if old_facility is not None:
            if self.mlg.has_edge(self, old_facility):
                edge = self.mlg.get_edge(self, old_facility)
                self.mlg.remove_edge(edge)
        if facility is not None:
            self.mlg.build_edge(self, facility, EdgeType.MEMBERSHIP)

    def set_ou(self, ou: Optional[Union[str, 'OrganizationalUnit']]):
        if ou is not None:
            ou = self.mlg.get_model_node(ou)
        old_ou = self.get_ou(False)
        self[OU] = ou
        if old_ou is not None:
            if self.mlg.has_edge(self, old_ou):
                edge = self.mlg.get_edge(self, old_ou)
                self.mlg.remove_edge(edge)
        self.mlg.build_edge(self, ou, EdgeType.MEMBERSHIP)

    def get(self, attribute_name: Any, default=None) -> Any:
        return self.attributes.get(attribute_name, default)

    def get_position_2d(self, create_default_position: bool = True):
        if not create_default_position and "pos" not in self.attributes:
            return None
        return self.attributes.get("pos", (0, 0))

    def get_position_3d(self, dimension: Optional[str] = None):
        """
        Returns the 3d position of this node.
        If no dimension is given, returns a tuple [x, y, z].
        If a dimension of "x", "y", or "z" is given, the respective dimension is returned.
        If dimension is set to "dict", a dict with the keys x, y, and z is returned.
        """
        pos2d = self.get_position_2d()
        pos3d = [pos2d[0], pos2d[1], self.layer.z + self.z_offset]
        if dimension is None:
            return pos3d
        pos3d = {
            "x": pos3d[0],
            "y": pos3d[1],
            "z": pos3d[2],
        }
        if dimension == "dict":
            return pos3d
        try:
            return pos3d[dimension]
        except KeyError:
            raise AttributeError(f"Invalid dimension: {dimension} can only be x, y, z, or dict")

    @property
    def pos3d(self):
        pos3d = self.get_position_3d()
        pos3d_namespace = types.SimpleNamespace()
        pos3d_namespace.x = pos3d[0]
        pos3d_namespace.y = pos3d[1]
        pos3d_namespace.z = pos3d[2]
        return pos3d_namespace

    def set_position_2d(self, pos_a: Union[float, Tuple[float]], y: Optional[float] = None):
        if isinstance(pos_a, Tuple):
            if len(pos_a) != 2:
                raise AttributeError("When using pos_a as position tuple, it has to be of length 2")
            self.attributes["pos"] = (pos_a[0], pos_a[1])
        else:
            if y is None:
                raise AttributeError("When using pos_a as the x-value, y has to be given")
            self.attributes["pos"] = (pos_a, y)

    def __str__(self):
        return f'"ModelNode {self.uid} ({self.name})"'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, ModelNode):
            return self.id == other.id
        return False

    def __lt__(self, other):
        if isinstance(other, ModelNode):
            return self.id < other.id
        return False

    def __gt__(self, other):
        if isinstance(other, ModelNode):
            return self.id > other.id
        return False

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def get_color(self):
        return self.layer.color if self.color is None else self.color

    def belongs_to_layer(self, layer: 'GraphLayer') -> bool:
        """
        Recursively checks if this node belongs to a layer, i.e., if the given layer is a parent layer of this node or
        the direct layer of this node.
        """
        if self.layer == layer:
            return True
        return layer.has_sub_layer(self.layer)

    def get_grid_element(self):
        element = self.get("element")
        if element is None or not isinstance(element, GridElement):
            raise ValueError("GridElement is not present")
        return element

    def get_description_lines(self) -> List[str]:
        return []
