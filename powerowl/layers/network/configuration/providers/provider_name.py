import enum


class ProviderName(enum.Enum):
    POWER_GRID = "power-grid"
    PANDAPOWER = "pandapower"
    MODBUS = "modbus"
    REGISTER = "register"
    PROTECTION = "protection"
