import dataclasses
from typing import List

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.layers.world.street import Street
from powerowl.layers.world.world_entity import WorldEntity


@dataclasses.dataclass(eq=False, kw_only=True)
class StreetNode(WorldEntity):
    """
    A street node marks points connected by one or multiple street segments.
    I.e., a street node is either a junction or a point on a single street.
    """
    def get_street(self) -> Street:
        streets = self.get_streets()
        if len(streets) == 0:
            raise ModelNodeNotFoundException("No Street for StreetNode found")
        return streets[0]

    def get_streets(self) -> List[Street]:
        # Find all associated neighbors of type street
        return []
