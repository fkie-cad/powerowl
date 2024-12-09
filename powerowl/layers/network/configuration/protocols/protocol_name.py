import enum


class ProtocolName(enum.Enum):
    IEC104 = "60870-5-104"
    MODBUS_TCP = "MODBUS/TCP"
    IEC61850_GOOSE = "61850-GOOSE"
    IEC61850_MMS = "61850-MMS"
