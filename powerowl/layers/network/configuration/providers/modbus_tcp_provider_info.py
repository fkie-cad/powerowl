from typing import Optional, Union

from powerowl.exceptions import DerivationError
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.protocols.modbus_tcp_info import ModbusTCPInfo
from powerowl.layers.network.configuration.providers.endian import Endian
from powerowl.layers.network.configuration.providers.modbus_types import ModbusTypes
from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
from powerowl.layers.network.configuration.providers.provider_name import ProviderName
from powerowl.layers.network.field_device import FieldDevice


class ModbusTCPProviderInfo(ProviderInfo):
    def __init__(self):
        super().__init__(ProviderName.MODBUS)
        self.register_address: int = 0
        self.table_offset: int = 0
        self.pdu_address: int = 0
        self.field_device: Optional[FieldDevice] = None
        self.data_type: ModbusTypes = ModbusTypes.INT32
        self.unit_id: int = 0
        self.endian: Endian = Endian.BIG_ENDIAN
        self.width_registers: int = 1
        self.polling_interval: int = 0

    def get_provider_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "address": self.register_address,
            "table_offset": self.table_offset,
            "zero_based_pdu_address": self.pdu_address,
            "device_id": self.field_device.uid,
            "type_id": self.data_type.value,
            "unit_id": self.unit_id,
            "polling_interval": self.polling_interval,
            "polling_enabled": self.polling_interval > 0,
            "width_registers": self.width_registers,
            "endian": self.endian.value,
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "address": self.register_address,
            "table_offset": self.table_offset,
            "zero_based_pdu_address": self.pdu_address,
            "field_device": self.field_device,
            "data_type": self.data_type,
            "unit_id": self.unit_id,
            "polling_interval": self.polling_interval,
            "endian": self.endian.value,
            "width_registers": self.width_registers,
        })
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.register_address = d["address"]
        self.table_offset = d["table_offset"]
        self.pdu_address = d["pdu_address"]
        self.field_device = d["field_device"]
        self.data_type = d["data_type"]
        self.unit_id = d["unit_id"]
        self.polling_interval = d["polling_interval"]
        self.endian = Endian(d["endian"])
        self.width_registers = d["width_registers"]

    @staticmethod
    def from_data_point(data_point: DataPoint) -> ProviderInfo:
        if not isinstance(data_point.protocol, ModbusTCPInfo):
            raise DerivationError(f"Invalid protocol: {data_point.protocol.name}")
        provider_info = ModbusTCPProviderInfo()
        protocol = data_point.protocol
        provider_info.register_address = protocol.address
        provider_info.domain = "source" if protocol.direction == "monitoring" else "target"
        provider_info.data_type = protocol.data_type
        provider_info.field_device = protocol.field_device
        provider_info.unit_id = protocol.unit_id
        provider_info.endian = protocol.endian
        provider_info.width_registers = protocol.width_registers
        provider_info.polling_interval = protocol.polling_interval
        provider_info.table_offset = protocol.table_offset
        provider_info.pdu_address = protocol.pdu_address
        return provider_info
