import abc
from abc import ABC
from typing import Optional, Set, TYPE_CHECKING, List

from powerowl.layers.network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo
from powerowl.layers.powergrid.elements.grid_element import GridElement
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext

if TYPE_CHECKING:
    from powerowl.layers.powergrid.elements.bus import Bus


class GridEdge(GridElement, ABC):
    """
    A grid element that connects two grid nodes (i.e., busses) and serves as an edge.
    """
    prefix: str = "grid-edge"

    def get_bus_a(self):
        return self.get_from_bus()

    def get_bus_b(self):
        return self.get_to_bus()

    def get_from_bus(self):
        return self.get_property_value("from_bus")

    def get_to_bus(self):
        return self.get_property_value("to_bus")

    def get_providers(self, filtered: bool = False, busses: Optional[Set['Bus']] = None) -> List[PowerGridProviderInfo]:
        providers = super().get_providers(filtered=filtered)
        if busses is None:
            return providers
        if len(busses) == 0:
            return []
        return self._filter_providers(providers, busses)

    def attribute_belongs_to_bus(self, bus: 'Bus', context: GridValueContext, attribute: str) -> bool:
        from_bus = self.get_from_bus()
        from_identifiers = ["from", "hv"]
        to_bus = self.get_to_bus()
        to_identifiers = ["to", "lv"]

        def _matches_identifiers(_attribute: str, _identifiers: List[str]) -> bool:
            for i in _identifiers:
                if f"_{i}_" in _attribute or _attribute.endswith(f"_{i}"):
                    return True
            return False

        matches_from = _matches_identifiers(attribute, from_identifiers)
        matches_to = _matches_identifiers(attribute, to_identifiers)

        if bus == from_bus:
            if matches_from:
                return True
            if not matches_to:
                return True
            return False
        elif bus == to_bus:
            if matches_to:
                return True
            if not matches_from:
                return True
            return False
        return False

    def _filter_providers(self,
                          providers: List[PowerGridProviderInfo],
                          sides: Set['Bus']) -> List[PowerGridProviderInfo]:
        filtered_providers: List[PowerGridProviderInfo] = []
        for provider in providers:
            for bus in sides:
                if self.attribute_belongs_to_bus(bus=bus,
                                                 context=provider.attribute_context,
                                                 attribute=provider.attribute_name):
                    filtered_providers.append(provider)
                    break
        return filtered_providers
