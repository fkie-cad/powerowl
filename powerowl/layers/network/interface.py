import dataclasses
import warnings
from typing import Optional, TYPE_CHECKING, List, cast

from .network_entity import NetworkEntity
from .network_node import NetworkNode
from ...graph.enums import Layers
from ...graph.enums.layer_matching_strategy import LayerMatchingStrategy

if TYPE_CHECKING:
    from .subnet import Subnet
    from .link import Link


@dataclasses.dataclass(eq=False, kw_only=True)
class Interface(NetworkEntity):
    mac_address: Optional[str] = None
    ip_address: Optional[str] = None
    short_name: str = None
    local_id: str = None

    def get_description_lines(self) -> List[str]:
        return [
            f"MAC: {self.mac_address}",
            f"IP:  {self.ip_address}",
            f"Subnet: {self.get_subnet().name if self.get_subnet() is not None else 'None'}"
        ]

    def get_local_id(self) -> str:
        if self.local_id is None:
            self.get_network_node().set_local_interface_ids()
        return self.local_id

    def get_global_short_id(self):
        return f"{self.get_network_node().uid}.{self.get_local_id()}"

    def get_subnet(self) -> Optional['Subnet']:
        from powerowl.layers.network.subnet import Subnet
        if not self.mlg.has_layer(Layers.SUBNETS):
            return None
        subnet_layer = self.mlg.get_layer(Layers.SUBNETS)
        subnets = self.mlg.get_inter_layer_neighbors(
            self,
            other_layers={subnet_layer},
            other_layers_matching_strategy=LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER,
            neighbor_instance_filter={Subnet}
        )
        if len(subnets) == 0:
            return None
        if len(subnets) > 1:
            warnings.warn(f"Interface {self.id} has multiple subnets assigned")
        return cast(Subnet, subnets[0])

    def get_network_node(self) -> Optional[NetworkNode]:
        neighbors = self.mlg.get_intra_layer_neighbors(
            self,
            neighbor_instance_filter={NetworkNode}
        )
        if len(neighbors) == 0:
            return None
        if len(neighbors) > 1:
            warnings.warn(f"Interface {self.id} has multiple nodes assigned")
        return cast(NetworkNode, neighbors[0])

    def get_network_link(self) -> Optional['Link']:
        from .link import Link
        neighbors = self.mlg.get_intra_layer_neighbors(
            self,
            neighbor_instance_filter={Link}
        )
        if len(neighbors) == 0:
            return None
        if len(neighbors) > 1:
            warnings.warn(f"Interface {self.id} has multiple links assigned")
        return cast(Link, neighbors[0])
