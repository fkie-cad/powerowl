import dataclasses

from .network_node import NetworkNode


@dataclasses.dataclass(eq=False, kw_only=True)
class Switch(NetworkNode):
    pass
