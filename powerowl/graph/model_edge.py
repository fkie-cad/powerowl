import dataclasses
import hashlib
from typing import TYPE_CHECKING

from powerowl.graph.enums import EdgeType
from powerowl.graph.enums.layer_matching_strategy import LayerMatchingStrategy

if TYPE_CHECKING:
    from powerowl.graph.model_node import ModelNode
    from powerowl.graph.multi_layer_graph import MultiLayerGraph


@dataclasses.dataclass
class ModelEdge:
    node_a: 'ModelNode' = None
    node_b: 'ModelNode' = None
    edge_type: EdgeType = EdgeType.DEFAULT

    def __post_init__(self):
        from powerowl.power_owl import PowerOwl
        self.id: int = PowerOwl.next_global_id()

    def to_dict(self):
        return {
            "id": self.id,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "edge_type": self.edge_type
        }

    def from_dict(self, d: dict):
        self.id = d["id"]
        self.node_a = d["node_a"]
        self.node_b = d["node_b"]
        self.edge_type = d["edge_type"]

    @property
    def mlg(self) -> 'MultiLayerGraph':
        return self.node_a.mlg

    @property
    def uid(self):
        return self.id

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
        return f'"{self.node_a.uid}<-{self.get_short_type()}@-{self.id}->{self.node_a.uid}"'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return int.from_bytes(hashlib.sha256(str(self).encode("utf-8")).digest(), byteorder="little")

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
