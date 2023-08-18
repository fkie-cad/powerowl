import enum


class SwitchType(enum.Enum):
    NONE = "None"
    LOAD_SWITCH = "LS"
    CIRCUIT_BREAKER = "CB"
    LOAD_BREAK_SWITCH = "LBS"
    DISCONNECTING_SWITCH = "DC"
