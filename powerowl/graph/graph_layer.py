from typing import Optional, TYPE_CHECKING, Union, List, Set, Type, Tuple, Callable

import networkx as nx

from .enums import EdgeType
from .enums.layer_matching_strategy import LayerMatchingStrategy
from .model_edge import ModelEdge
from .model_node import ModelNode
from ..exceptions.layer_not_found_exception import LayerNotFoundException

if TYPE_CHECKING:
    from .multi_layer_graph import MultiLayerGraph


class GraphLayer:
    _gid: int = 0

    def __init__(self, mlg: 'MultiLayerGraph' = None, name: str = None, z_offset: int = 0,
                 parent: Optional['GraphLayer'] = None):
        self.mlg = mlg
        self._name = name
        self.color = "#000000"
        self.z_offset = z_offset
        self.uid = GraphLayer._gid
        self.parent: Optional[GraphLayer] = parent
        GraphLayer._gid += 1

    @property
    def name(self):
        if self.parent is None:
            return self._name
        return f"{self.parent.name}.{self._name}"

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def solo_name(self):
        return self._name

    @property
    def z(self) -> float:
        base = 0 if self.parent is None else self.parent.z
        return base + self.z_offset

    def get_layer_graph(self, include_sub_layers: bool = True, edge_type_filter: Optional[Set[EdgeType]] = None) -> nx.Graph:
        """
        From the Multi Layer Graph, this method calculates a one-layered graph containing only nodes and edges from
        this layer. If specified, sub-layers are included as well.
        Inter-Layer edges are dropped.
        Note: While the graph is copied, node-attributes are identical, i.e., potentially references to the MLG nodes.
        """
        # Create a list of layers that should be included
        allowed_layers: Set['GraphLayer'] = {self}
        if include_sub_layers:
            allowed_layers.union(self.get_sub_layers())
        # Use the MultiLayerGraph to calculate the subgraph
        return self.mlg.get_subgraph_of_layers(allowed_layers, edge_type_filter=edge_type_filter)

    def get_lower_layer(self) -> Optional['GraphLayer']:
        layers = self.mlg.get_layers()
        my_index = layers.index(self)
        if my_index > 0:
            return layers[my_index - 1]
        return None

    def get_higher_layer(self) -> Optional['GraphLayer']:
        layers = self.mlg.get_layers()
        my_index = layers.index(self)
        if my_index + 1 < len(layers):
            return layers[my_index + 1]
        return None

    def has_lower_layer(self):
        return self.get_lower_layer() is not None

    def has_higher_layer(self):
        return self.get_higher_layer() is not None

    def add_sub_layer(self, name: str, z_offset: float = 0) -> 'GraphLayer':
        return self.mlg.add_layer(name, z_offset=z_offset, parent=self)

    def get_sub_layers(self, recursive: bool = False) -> List['GraphLayer']:
        sub_layers = {layer for layer in self.mlg.get_layers() if layer.parent == self}
        if recursive:
            for layer in sub_layers:
                sub_layers.update(layer.get_sub_layers(recursive=recursive))
        return sorted(sub_layers)

    def get_main_layer(self) -> 'GraphLayer':
        if self.parent is None:
            return self
        return self.parent.get_main_layer()

    def has_sub_layer(self, layer: 'GraphLayer', transitive: bool = True):
        """
        Checks whether the given layer is a sub-layer of this layer.
        If transitive is True, transitive sub-layers are considered as well.
        """
        sub_layers = self.get_sub_layers(recursive=transitive)
        return self == layer or layer in sub_layers

    def is_sub_layer(self, layer: 'GraphLayer', transitive: bool = True):
        """
        Checks whether this layer is a sub-layer of the given layer.
        If transitive is True, transitive sub-layers are considered as well.
        """
        return layer.has_sub_layer(self, transitive=transitive)

    def has_sub_layer_relation(self, layer: 'GraphLayer', transitive: bool = True):
        """
        Checks whether this layer has a sub-layer relation with the given layer.
        If transitive is True, transitive relations are considered as well.
        """
        return self.has_sub_layer(layer, transitive=transitive) or self.is_sub_layer(layer, transitive=transitive)

    def get_sub_layer_name(self, n):
        model_node = self.mlg.get_model_node(n)
        return model_node.layer.name

    def get_nodes(self,
                  layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.IS_SUB_LAYER
                  ) -> List['ModelNode']:
        return self.mlg.get_nodes_at_layers({self}, layer_matching_strategy=layer_matching_strategy)

    def get_sub_layer_of_edge(self, e, selection: str = "lowest"):
        sub_layer_a = self.get_sub_layer_name(e[0])
        z_a = self.get_sub_layer_z(sub_layer_a)
        sub_layer_b = self.get_sub_layer_name(e[1])
        z_b = self.get_sub_layer_z(sub_layer_b)
        if z_a < z_b and selection == "lowest":
            return sub_layer_a
        elif z_a > z_b and selection == "highest":
            return sub_layer_a
        return sub_layer_b

    def get_sub_layer(self, sub_layer_name: str) -> 'GraphLayer':
        for sub_layer in self.get_sub_layers():
            if sub_layer.name == sub_layer_name:
                return sub_layer
        raise LayerNotFoundException(f"No sub-layer with name {sub_layer_name}")

    def get_sub_layer_z(self, sub_layer_name):
        return self.get_sub_layer(sub_layer_name).z

    def add_node(self, model_node: ModelNode):
        model_node.layer = self
        return self.mlg.add_node(model_node=model_node)

    def add_edge(self, node_a: Union[str, 'ModelNode'], node_b: Union[str, 'ModelNode'],
                 edge_type: EdgeType = EdgeType.DEFAULT):
        node_a = self.mlg.get_model_node(node_a)
        node_b = self.mlg.get_model_node(node_b)
        edge = ModelEdge(node_a=node_a, node_b=node_b, edge_type=edge_type)
        return self.mlg.add_edge(edge)

    def __str__(self):
        return self.name

    def __lt__(self, other):
        if not isinstance(other, GraphLayer):
            return False
        return self.z < other.z

    def __le__(self, other):
        if not isinstance(other, GraphLayer):
            return False
        return self.z <= other.z

    def __eq__(self, other):
        if not isinstance(other, GraphLayer):
            return False
        return self.uid == other.uid

    def __ge__(self, other):
        if not isinstance(other, GraphLayer):
            return False
        return self.z >= other.z

    def __gt__(self, other):
        if not isinstance(other, GraphLayer):
            return False
        return self.z > other.z

    def __hash__(self):
        return self.uid

    def to_dict(self) -> dict:
        return {
            "name": self._name,
            "z_offset": self.z_offset,
            "color": self.color,
            "uid": self.uid,
            "parent": self.parent
        }

    def from_dict(self, d: dict):
        self.name = d["name"]
        self.z_offset = d["z_offset"]
        self.color = d["color"]
        self.uid = d["uid"]
        self.parent = d["parent"]
