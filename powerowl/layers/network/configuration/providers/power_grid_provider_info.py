from typing import Optional, Type, Any, TYPE_CHECKING

from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
from powerowl.layers.network.configuration.providers.provider_name import ProviderName
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext

if TYPE_CHECKING:
    from powerowl.layers.powergrid.values.grid_value import GridValue
    from powerowl.layers.powergrid import PowerGridModel


class PowerGridProviderInfo(ProviderInfo):
    def __init__(self):
        super().__init__(ProviderName.POWER_GRID)
        self.element_id: Optional[str] = None
        self.attribute_context: GridValueContext = GridValueContext.GENERIC
        self.attribute_name: Optional[str] = None
        self.attribute_type: Type = Any

    def get_provider_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "grid_element": self.element_id,
            "context": self.attribute_context.name,
            "attribute": self.attribute_name,
            "type": str(self.attribute_type.__name__)
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["grid_element"] = self.element_id
        d["attribute"] = self.attribute_name
        d["context"] = self.attribute_context
        d["type"] = self.attribute_type
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.element_id = d["grid_element"]
        self.attribute_context = d["context"]
        self.attribute_name = d["attribute"]
        self.attribute_type = d["type"]

    def get_grid_value(self, grid_model: 'PowerGridModel') -> 'GridValue':
        return grid_model.get_element_by_identifier(self.element_id).get(self.attribute_name, self.attribute_context)
