from typing import List, Union

import numpy as np
from powerowl.layers.network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo

from . import GridElement
from .attribute_specification import AttributeSpecification as As
from .enums.switch_type import SwitchType
from .grid_annotator import GridAnnotator
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit
from ...network.configuration.providers.protection_event_provider_info import ProtectionEventProviderInfo
from ...network.configuration.providers.provider_info import ProviderInfo


class Switch(GridAnnotator):
    prefix = "switch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),
            As("controllable", Gvc.PROPERTY, bool, True, required=False),
            As("current_protection_enabled", Gvc.PROPERTY, bool, False, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("element", Gvc.PROPERTY, GridElement, None),
            As("type", Gvc.PROPERTY, SwitchType, SwitchType.NONE),
            As("maximum_current", Gvc.PROPERTY, float, np.NAN, Unit.AMPERE, Scale.BASE, required=False),

            As("in_service", Gvc.CONFIGURATION, bool, True, required=False, operator_controllable=False, simulation_context=False),
            As("closed", Gvc.CONFIGURATION, bool, True, related=[(Gvc.MEASUREMENT, "is_closed")]),

            As("is_closed", Gvc.MEASUREMENT, bool, True, source=(Gvc.CONFIGURATION, "closed"), targets=[(Gvc.CONFIGURATION, "closed")])
        ]

    def et(self):
        return self.get_property("element").value.prefix

    def get_providers(self, filtered: bool = False) -> List[ProviderInfo]:
        providers = []
        if self.get_property_value("current_protection_enabled"):
            provider = ProtectionEventProviderInfo()
            provider.domain = "source"
            provider.element_id = self.get_identifier()
            provider.protection_event = "current_over_load"
            providers.append(provider)
        providers.extend(super().get_providers(filtered))
        return providers
