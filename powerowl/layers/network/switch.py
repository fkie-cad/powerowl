import dataclasses
from typing import Optional

from .network_node import NetworkNode


@dataclasses.dataclass(eq=False, kw_only=True)
class Switch(NetworkNode):
    def __post_init__(self):
        super().__post_init__()
        self.is_manageable = False

    @property
    def is_manageable(self) -> bool:
        return self.get_configuration_key("is_manageable", False)

    @is_manageable.setter
    def is_manageable(self, is_manageable: bool) -> None:
        self.set_configuration_key("is_manageable", is_manageable)

    @property
    def management_ip_address(self) -> Optional[str]:
        return self.get_configuration_key("management_ip_address", None)

    @management_ip_address.setter
    def management_ip_address(self, management_ip_address: Optional[str]) -> None:
        self.set_configuration_key("management_ip_address", management_ip_address)
