import dataclasses
import warnings
from typing import List, cast

from .interface import Interface
from .link_type import LinkType
from .network_entity import NetworkEntity
from ...graph.enums import Layers
from ...graph.enums.layer_matching_strategy import LayerMatchingStrategy


@dataclasses.dataclass(eq=False, kw_only=True)
class Link(NetworkEntity):
    bandwidth: str = "1Gbps"
    delay: str = "2ms"
    jitter: str = ""
    packet_loss: str = "0%"
    bit_error_rate: str = "0%"
    link_type: LinkType = LinkType.DIGITAL

    def get_interfaces(self) -> List['Interface']:
        neighbors = self.mlg.get_intra_layer_neighbors(
            self,
            neighbor_instance_filter={Interface}
        )
        if len(neighbors) != 2:
            warnings.warn(f"Link {self.id} has {len(neighbors)} interfaces, should be 2")
        return [cast(Interface, neighbor) for neighbor in neighbors]

    def get_link_properties(self) -> dict:
        return {
            "bandwidth": self.bandwidth,
            "data-rate": self.bandwidth,
            "delay": self.delay,
            "jitter": self.jitter,
            "packet-loss": self.packet_loss,
            "bit-error-rate": self.bit_error_rate,
            "link-type": self.link_type.value
        }
