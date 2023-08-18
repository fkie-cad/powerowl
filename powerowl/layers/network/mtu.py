import dataclasses
from typing import cast

from powerowl.layers.network.host import Host


@dataclasses.dataclass(eq=False, kw_only=True)
class MTU(Host):
    def get_rtus(self):
        from powerowl.layers.network.rtu import RTU
        return [cast(RTU, rtu) for rtu in self.layer.mlg.get_neighbors(self, {RTU})]
