import enum


class ProtocolName(enum.Enum):
    IEC104 = "60870-5-104"
    MODBUS_TCP = "MODBUS/TCP"
    GOOSE = "61850-GOOSE"
