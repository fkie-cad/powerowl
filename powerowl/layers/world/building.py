import dataclasses

from powerowl.layers.world.world_entity import WorldEntity


@dataclasses.dataclass(eq=False, kw_only=True)
class Building(WorldEntity):
    pass
