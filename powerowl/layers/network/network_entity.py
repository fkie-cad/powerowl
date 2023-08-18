import dataclasses

from ...graph.model_node import ModelNode


@dataclasses.dataclass(eq=False, kw_only=True)
class NetworkEntity(ModelNode):
    pass
