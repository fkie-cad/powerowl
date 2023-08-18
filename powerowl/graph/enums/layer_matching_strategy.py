import enum
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from powerowl.graph import GraphLayer


class LayerMatchingStrategy(enum.Enum):
    """
    The LayerSelectionStrategy defines how sub-layers are handled by various methods.
    EXACT: Exact: Only the exact layer is a match
    IS_SUB_LAYER: If the first layer is a (transitive) sub layer of the second layer (or identical)
    IS_PARENT: If the first layer is a (transitive) parent layer of the second layer (or identical)
    TRANSITIVE: If one of the layer is a sub layer of the other one (or identical)
    SHARED_TOP_LEVEL_LAYER: If both layers share the same top-level layer
    SIBLINGS: If both layers share the same parent
    ANY: Layers always match each other
    """
    EXACT = 0
    TRANSITIVE = 1
    IS_SUB_LAYER = 2
    IS_PARENT = 3
    SHARED_TOP_LEVEL_LAYER = 4
    SIBLINGS = 5
    ANY = 10

    def matches(self, layer_a: 'GraphLayer', layer_b: 'GraphLayer'):
        if self == LayerMatchingStrategy.ANY:
            return True
        if self == LayerMatchingStrategy.EXACT:
            return layer_a == layer_b
        if self == LayerMatchingStrategy.TRANSITIVE:
            return layer_a.has_sub_layer_relation(layer_b, transitive=True)
        if self == LayerMatchingStrategy.IS_SUB_LAYER:
            return layer_a.is_sub_layer(layer_b, transitive=True)
        if self == LayerMatchingStrategy.IS_PARENT:
            return layer_a.has_sub_layer(layer_b, transitive=True)
        if self == LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER:
            return layer_a.get_main_layer() == layer_b.get_main_layer()
        if self == LayerMatchingStrategy.SIBLINGS:
            return layer_a.parent == layer_b.parent
        warnings.warn(f"You forgot to implement an implementation for {self}")
        return False
