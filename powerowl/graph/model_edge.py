import dataclasses
import hashlib
from typing import TYPE_CHECKING, Callable, List, Set, Iterable

import shapely
from powerowl.graph.enums import EdgeType
from powerowl.graph.enums.layer_matching_strategy import LayerMatchingStrategy

if TYPE_CHECKING:
    from powerowl.graph.model_node import ModelNode
    from powerowl.graph.multi_layer_graph import MultiLayerGraph


@dataclasses.dataclass
class ModelEdge:
    node_a: 'ModelNode' = None
    node_b: 'ModelNode' = None
    edge_types: List[EdgeType] = dataclasses.field(default_factory=lambda: [EdgeType.DEFAULT])

    def __post_init__(self):
        from powerowl.power_owl import PowerOwl
        self.id: int = PowerOwl.next_global_id(self)
        self._options = {}
        self._on_set_id_callbacks: List[Callable[['ModelNode'], None]] = []

    def to_dict(self):
        return {
            "id": self.id,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "edge_types": self.edge_types,
            "options": self._options,
        }

    def get_option(self, option, default=None):
        return self._options.get(option, default)

    def clear_option(self, option):
        self._options.pop(option, None)

    def set_option(self, option, value):
        self._options[option] = value

    @property
    def length(self) -> int:
        """
        The length of the edge w.r.t. the 2-dimensional positions of the associated nodes.
        :return: The length of the edge.
        """
        length = 0
        pos_a = self.node_a.get_position_2d(create_default_position=False)
        pos_b = self.node_b.get_position_2d(create_default_position=False)
        if pos_a is not None and pos_b is not None:
            length = shapely.distance(shapely.Point(pos_a), shapely.Point(pos_b))
        return length

    def from_dict(self, d: dict):
        self.id = d["id"]
        self.node_a = d["node_a"]
        self.node_b = d["node_b"]
        self.edge_types = d["edge_types"]
        self._options = d.get("options", {})

    @property
    def edge_type(self) -> EdgeType:
        """
        Returns the primary type of this edge. If no type is assigned, EdgeType.DEFAULT is returned.

        """
        if len(self.edge_types) == 0:
            return EdgeType.DEFAULT
        return self.edge_types[0]

    def add_type(self, edge_type: EdgeType) -> bool:
        if self.has_type(edge_type):
            return False
        self.edge_types.append(edge_type)
        return True

    def remove_type(self, edge_type: EdgeType) -> bool:
        if self.has_type(edge_type):
            self.edge_types.remove(edge_type)
            return True
        return False

    def has_type(self, edge_type: EdgeType) -> bool:
        return edge_type in self.edge_types

    def has_any_type(self, edge_types: Iterable[EdgeType]) -> bool:
        for edge_type in edge_types:
            if self.has_type(edge_type):
                return True
        return False

    def has_all_types(self, edge_types: Iterable[EdgeType]) -> bool:
        for edge_type in edge_types:
            if not self.has_type(edge_type):
                return False
        return True

    @property
    def mlg(self) -> 'MultiLayerGraph':
        return self.node_a.mlg

    @property
    def uid(self) -> id:
        return self.id

    def set_id(self, id: int):
        self.id = id
        for on_set_id in self._on_set_id_callbacks:
            on_set_id(self)

    def add_on_set_id_callback(self, callback: Callable[['ModelNode'], None]):
        self._on_set_id_callbacks.append(callback)

    def is_inter_layer_edge(self,
                            layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.TRANSITIVE
                            ) -> bool:
        """
        Checks whether this edge spans multiple layers.
        The layer_matching_strategy defines how layers are distinguished, i.e., only edges that span two layers that
            are not matched by the layer_matching_strategy are included in the result.
        """
        layer_a = self.node_a.layer
        layer_b = self.node_b.layer
        return not layer_matching_strategy.matches(layer_a, layer_b)

    def get_short_type(self) -> str:
        # TODO: Use actual type
        return self.edge_type.get_short_name()

    def __str__(self):
        return f'"{self.node_a.uid}<-{self.get_short_type()}@-{self.id}->{self.node_b.uid}"'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return int.from_bytes(hashlib.sha256(str(self.id).encode("utf-8")).digest(), byteorder="little")

    def __eq__(self, other):
        if isinstance(other, ModelEdge):
            return self.id == other.id
        return False

    def __lt__(self, other):
        if isinstance(other, ModelEdge):
            return self.id < other.id
        return False

    def __gt__(self, other):
        if isinstance(other, ModelEdge):
            return self.id > other.id
        return False

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other
