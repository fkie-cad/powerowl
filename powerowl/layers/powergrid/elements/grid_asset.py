from abc import ABC
from typing import Optional, List
import numpy as np

from .bus import Bus
from .enums.voltage_niveau import VoltageNiveau
from .grid_element import GridElement


class GridAsset(GridElement, ABC):
    """
    A grid element that belongs to a single grid node (i.e., bus)
    """
    prefix: str = "grid-asset"

    def get_bus(self) -> 'Bus':
        return self.get_property_value("bus")

    def get_voltage_niveau(self) -> VoltageNiveau:
        return VoltageNiveau.NONE

    def get_buses(self) -> List['Bus']:
        return [self.get_bus()]

    def get_maximum_power(self, dimension) -> Optional[float]:
        max_power = self.get_property_value(f"maximum_{dimension}_power", None)
        nominal_power = self.get_property_value("nominal_power", None)
        if max_power is None or np.isnan(max_power):
            if nominal_power is None or np.isnan(nominal_power):
                return None
            # Fallback to nominal power
            max_power = nominal_power
        return max_power

    def get_nominal_power(self) -> Optional[float]:
        max_power = self.get_property_value(f"maximum_active_power", None)
        nominal_power = self.get_property_value("nominal_power", None)
        if nominal_power is None or np.isnan(nominal_power):
            return max_power
        return nominal_power

    def get_maximum_active_power(self) -> Optional[float]:
        return self.get_maximum_power("active")

    def get_maximum_reactive_power(self) -> Optional[float]:
        return self.get_maximum_power("reactive")
