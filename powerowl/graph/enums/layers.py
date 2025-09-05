import enum


class Layers(str, enum.Enum):
    POWER_GRID = "power-grid"
    POWER_GRID_ASSETS = "power-grid-assets"
    POWER_GRID_CORE = "power-grid-core"
    FACILITIES = "facilities"
    OUs = "ous"
    NETWORK = "network"
    NETWORK_BACKBONE = "network-backbone"
    NETWORK_LAN = "network-lan"
    NETWORK_FIELD = "network-field"
    NETWORK_INTERNET = "network-internet"

    SAFETY = "safety"
    DATA_POINTS = "data-points"
    DATA_POINTS_IEC104 = "data-points-iec104"
    DATA_POINTS_IEC61850 = "data-points-61850"
    DATA_POINTS_MODBUS = "data-points-modbus"
    SUBNETS = "subnets"

    WORLD = "world"
    WORLD_STREETS = "world-street"
    WORLD_BUILDINGS = "world-buildings"
    WORLD_STREET_SEGMENTS = "world-street-segments"
