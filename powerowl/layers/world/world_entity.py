import dataclasses

from powerowl.graph.model_node import ModelNode


@dataclasses.dataclass(eq=False, kw_only=True)
class WorldEntity(ModelNode):
    def __str__(self):
        return f'"WorldEntity: {self.name}"'
