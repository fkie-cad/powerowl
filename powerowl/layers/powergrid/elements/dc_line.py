from typing import List, Set

from .attribute_specification import AttributeSpecification as As
from .grid_edge import GridEdge
from .bus import Bus
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit
from ...network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo


class DcLine(GridEdge):
    prefix = "dcline"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("from_bus", Gvc.PROPERTY, Bus, None),
            As("to_bus", Gvc.PROPERTY, Bus, None),
            As("loss_relative", Gvc.PROPERTY, float, None, Unit.PERCENT),
            As("loss_total", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_reactive_power_from", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("maximum_reactive_power_from", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("minimum_reactive_power_to", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("maximum_reactive_power_to", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),

            As("in_service", Gvc.CONFIGURATION, bool, True),
            As("active_power", Gvc.CONFIGURATION, float, None, Unit.WATT, Scale.BASE),
            As("voltage_from_bus", Gvc.CONFIGURATION, float, None, Unit.PER_UNIT),
            As("voltage_to_bus", Gvc.CONFIGURATION, float, None, Unit.PER_UNIT),

            As("active_power_from", Gvc.MEASUREMENT, float, None, Unit.WATT, Scale.BASE, required=False),
            As("reactive_power_from", Gvc.MEASUREMENT, float, None, Unit.VAR, Scale.BASE, required=False),
            As("active_power_to", Gvc.MEASUREMENT, float, None, Unit.WATT, Scale.BASE, required=False),
            As("reactive_power_to", Gvc.MEASUREMENT, float, None, Unit.VAR, Scale.BASE, required=False),
            As("active_power_loss", Gvc.MEASUREMENT, float, None, Unit.WATT, Scale.BASE, required=False),
            As("voltage_from_bus", Gvc.MEASUREMENT, float, None, Unit.PER_UNIT, required=False),
            As("voltage_to_bus", Gvc.MEASUREMENT, float, None, Unit.PER_UNIT, required=False),
            As("voltage_angle_from_bus", Gvc.MEASUREMENT, float, None, Unit.DEGREE, required=False),
            As("voltage_angle_to_bus", Gvc.MEASUREMENT, float, None, Unit.DEGREE, required=False),
        ]



