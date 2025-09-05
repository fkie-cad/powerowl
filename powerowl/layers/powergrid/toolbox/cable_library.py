from typing import Optional

import shapely
from shapely import Point

from powerowl.layers.powergrid.elements import Bus, Line
from powerowl.layers.powergrid.elements.enums.line_type import LineType
from powerowl.layers.powergrid.values.units.scale import Scale


class CableLibrary:
    @staticmethod
    def create_house_attachment_cable_nyy_j_6mm(from_bus: Bus, to_bus: Bus, name: Optional[str] = None, length: Optional[float] = None) -> Line:
        """
        Creates a Line connecting the two given buses.
        This line represents a multi-phase cable, e.g., an NYY-J cable, for 0.4-1kV used for attaching a single household.
        Args:
            from_bus: The first bus
            to_bus: The second bus
            name: The name of the line
            length: The length of the line. If not given, the buses' geo_positions will be used.

        Returns:
            The line connecting the two given buses.
        """
        if name is None:
            name = f"House-Connection-{from_bus.index}-{to_bus.index}"
        if length is None:
            bus_pos_a = Point(from_bus.get_property_value("geo_position"))
            bus_pos_b = Point(to_bus.get_property_value("geo_position"))
            length = shapely.distance(bus_pos_a, bus_pos_b)
        return Line(
            name=name,
            from_bus=from_bus,
            to_bus=to_bus,
            type=LineType.UNDERGROUND_CABLE,
            length=length,
            maximum_current=24,
            resistance_per_km=3.08,
            reactance_per_km=0.138,
            capacitance_per_km=Scale.NANO.to_base(8),
        )

    @staticmethod
    def _create_lv_street_cable_nayy_j_x(area_mm2: float, from_bus: Bus, to_bus: Bus, name: Optional[str] = None, length: Optional[float] = None) -> Line:
        if name is None:
            name = f"Street-Cable-{from_bus.index}-{to_bus.index}"
        if length is None:
            bus_pos_a = Point(from_bus.get_property_value("geo_position"))
            bus_pos_b = Point(to_bus.get_property_value("geo_position"))
            length = shapely.distance(bus_pos_a, bus_pos_b)

        specs = dict(
            max_current=30,
            resistance_per_km=2,
            reactance_per_km=0.1,
            capacitance_per_km=Scale.NANO.to_base(10)
        )
        defaults = {
            "150": {
                "max_current": 270,
                "resistance_per_km": 0.208,
                "reactance_per_km": 0.08,
                "capacitance_per_km": Scale.NANO.to_base(261),
            },
            "240": {
                "max_current": 375,
                "resistance_per_km": 0.125,
                "reactance_per_km": 0.08,
                "capacitance_per_km": Scale.NANO.to_base(270),
            },
            "350": {
                "max_current": 450,
                "resistance_per_km": 0.082,
                "reactance_per_km": 0.08,
                "capacitance_per_km": Scale.NANO.to_base(270),
            }
        }

        if str(area_mm2) in defaults:
            specs.update(defaults[str(area_mm2)])

        return Line(
            name=name,
            from_bus=from_bus,
            to_bus=to_bus,
            type=LineType.UNDERGROUND_CABLE,
            length=length,
            maximum_current=specs["max_current"],
            resistance_per_km=specs["resistance_per_km"],
            reactance_per_km=specs["reactance_per_km"],
            capacitance_per_km=specs["capacitance_per_km"],
        )

    @staticmethod
    def create_lv_street_cable_nayy_j_150(from_bus: Bus, to_bus: Bus, name: Optional[str] = None, length: Optional[float] = None) -> Line:
        return CableLibrary._create_lv_street_cable_nayy_j_x(150, from_bus, to_bus, name, length)

    @staticmethod
    def create_lv_street_cable_nayy_j_240(from_bus: Bus, to_bus: Bus, name: Optional[str] = None, length: Optional[float] = None) -> Line:
        return CableLibrary._create_lv_street_cable_nayy_j_x(240, from_bus, to_bus, name, length)

    @staticmethod
    def create_lv_street_cable_nayy_j_350(from_bus: Bus, to_bus: Bus, name: Optional[str] = None, length: Optional[float] = None) -> Line:
        return CableLibrary._create_lv_street_cable_nayy_j_x(350, from_bus, to_bus, name, length)
