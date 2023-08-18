import enum


class StaticGeneratorType(enum.Enum):
    NONE = "none"
    WIND = "WP"
    PHOTOVOLTAIC = "PV"
    COMBINED_HEATING_AND_POWER = "CHP"
    COMBINED_HEATING_AND_POWER_DIESEL = "CHPD"
    FUEL_CELL = "FC"
    FUEL_CELL_RESIDENTIAL = "RFC"
