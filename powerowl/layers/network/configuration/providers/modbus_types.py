import enum
from typing import Type


class ModbusTypes(enum.Enum):
    INT32 = "int32"
    FLOAT32 = "float32"
    BOOL = "bool"

    @staticmethod
    def from_native(attribute_type: Type) -> 'ModbusTypes':
        if attribute_type == float:
            return ModbusTypes.FLOAT32
        if attribute_type == int:
            return ModbusTypes.INT32
        if attribute_type == bool:
            return ModbusTypes.BOOL
        raise AttributeError(f"Invalid attribute type for Modbus: {attribute_type.__name__}")
