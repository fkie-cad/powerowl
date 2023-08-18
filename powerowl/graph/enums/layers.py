import enum


class Layers(str, enum.Enum):
    POWER_GRID: str = "power-grid"
    POWER_GRID_ASSETS: str = "power-grid-assets"
    POWER_GRID_CORE: str = "power-grid-core"
    FACILITIES: str = "facilities"
    OUs: str = "ous"
    NETWORK: str = "network"
    NETWORK_BACKBONE: str = "network-backbone"
    NETWORK_LAN: str = "network-lan"
    NETWORK_FIELD: str = "network-field"
    SAFETY: str = "safety"
    DATA_POINTS: str = "data-points"
    DATA_POINTS_IEC104: str = "data-points-iec104"
    DATA_POINTS_IEC61850: str = "data-points-61850"
    DATA_POINTS_MODBUS: str = "data-points-modbus"
    SUBNETS: str = "subnets"
