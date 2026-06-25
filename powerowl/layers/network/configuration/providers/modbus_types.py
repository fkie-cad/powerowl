import enum
from typing import Type

from numba import double
from powerowl.layers.network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo
from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
from powerowl.layers.powergrid.values.grid_value_type import Step


class ModbusTypes(enum.Enum):
    SHORT = "int16"         # Single register
    INT32 = "int32"         # Two registers
    INT64 = "int64"         # Four registers
    FLOAT32 = "float32"     # 32 Bit Float
    DOUBLE = "double"       # 64 Bit Float
    STEP = "step"           # Coil / Bit
    BOOL = "bool"           # Coil / Bit
    STRING = "string"       # Multiple registers, 2 chars per register

    @staticmethod
    def from_provider(provider: ProviderInfo):
        if isinstance(provider, PowerGridProviderInfo):
            return ModbusTypes.from_native(provider.attribute_type)
        raise AttributeError(f"Unsupported provider type {type(provider)} for Modbus")

    def to_native(self):
        if self in [ModbusTypes.BOOL]:
            return bool
        if self in [ModbusTypes.STEP]:
            return Step
        if self in [ModbusTypes.DOUBLE, ModbusTypes.FLOAT32]:
            return float
        if self in [ModbusTypes.SHORT, ModbusTypes.INT32, ModbusTypes.INT64]:
            return int
        if self in [ModbusTypes.STRING]:
            return str
        raise AttributeError(f"Unsupported modbus type {type(self)}")

    @staticmethod
    def from_native(attribute_type: Type) -> 'ModbusTypes':
        if attribute_type == float:
            return ModbusTypes.FLOAT32
        if attribute_type == int:
            return ModbusTypes.INT32
        if attribute_type == bool:
            return ModbusTypes.BOOL
        if attribute_type == Step:
            return ModbusTypes.STEP
        if attribute_type == str:
            return ModbusTypes.STRING

        raise AttributeError(f"Invalid attribute type for Modbus: {attribute_type.__name__}")
