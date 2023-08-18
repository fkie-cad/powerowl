import dataclasses

from .host import Host


@dataclasses.dataclass(eq=False, kw_only=True)
class Router(Host):
    pass
