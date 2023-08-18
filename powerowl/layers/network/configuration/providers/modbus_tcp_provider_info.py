from typing import Optional

from powerowl.exceptions import DerivationError
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.protocols.modbus_tcp_info import ModbusTCPInfo
from powerowl.layers.network.configuration.providers.modbus_types import ModbusTypes
from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
from powerowl.layers.network.configuration.providers.provider_name import ProviderName
from powerowl.layers.network.field_device import FieldDevice
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext


class ModbusTCPProviderInfo(ProviderInfo):
    def __init__(self):
        super().__init__(ProviderName.MODBUS)
        self.modbus_address: int = 0
        self.field_device: Optional[FieldDevice] = None
        self.data_type: ModbusTypes = ModbusTypes.INT32
        self.unit_id: int = 0

    def get_provider_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "address": self.modbus_address,
            "field_id": self.field_device.name,
            "type_id": self.data_type.value,
            "unit_id": self.unit_id
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "address": self.modbus_address,
            "field_device": self.field_device,
            "data_type": self.data_type,
            "unit_id": self.unit_id
        })
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.modbus_address = d["address"]
        self.field_device = d["field_device"]
        self.data_type = d["data_type"]
        self.unit_id = d["unit_id"]

    @staticmethod
    def from_data_point(data_point: DataPoint):
        if not isinstance(data_point.protocol, ModbusTCPInfo):
            raise DerivationError(f"Invalid protocol: {data_point.protocol.name}")
        provider_info = ModbusTCPProviderInfo()
        protocol = data_point.protocol
        provider_info.domain = "source" if protocol.direction == "monitoring" else "target"
        provider_info.data_type = protocol.data_type
        provider_info.field_device = protocol.field_device
        provider_info.unit_id = protocol.unit_id
        return provider_info
