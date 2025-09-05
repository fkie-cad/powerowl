from typing import Optional

from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
from powerowl.layers.network.configuration.providers.provider_name import ProviderName


class ProtectionEventProviderInfo(ProviderInfo):
    def __init__(self):
        super().__init__(ProviderName.PROTECTION)
        self.element_id: Optional[str] = None
        self.protection_event: Optional[str] = None

    def get_provider_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "grid_element": self.element_id,
            "protection_event": self.protection_event
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["grid_element"] = self.element_id
        d["protection_event"] = self.protection_event
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.element_id = d["grid_element"]
        self.protection_event = d["protection_event"]
