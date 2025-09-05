import dataclasses

from .network_node import NetworkNode


@dataclasses.dataclass(eq=False, kw_only=True)
class InternetGateway(NetworkNode):
    pass
