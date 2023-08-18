from typing import Optional, TYPE_CHECKING

from powerowl.layers.network.configuration.protocols.protocol_info import ProtocolInfo
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName
from powerowl.layers.network.configuration.providers.modbus_types import ModbusTypes

if TYPE_CHECKING:
    from powerowl.layers.network.field_device import FieldDevice


class ModbusTCPInfo(ProtocolInfo):
    def __init__(self):
        super().__init__(ProtocolName.MODBUS_TCP)
        self._direction = "monitoring"
        self._address: int = 0
        self.field_device: Optional['FieldDevice'] = None
        self.data_type: ModbusTypes = ModbusTypes.INT32
        self.unit_id: int = 0

    def generate_data_point_id(self):
        return f"{self.field_device.name}.{self.unit_id}.{self.address}.{self.table[0].lower()}"

    @property
    def table(self) -> str:
        if self.data_type == ModbusTypes.BOOL:
            return "c"
        return "r"

    @property
    def address(self):
        return self._address

    @address.setter
    def address(self, address: int):
        self._address = address

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction: str):
        direction = direction.lower()
        if direction not in ["monitoring", "control"]:
            raise ValueError(f"Unexpected direction {direction}, expecting 'monitoring' or 'control'")
        self._direction = direction

    @property
    def register_size_bits(self) -> int:
        return self.register_size_bytes * 8

    @property
    def register_size_bytes(self) -> int:
        return 2

    @property
    def width_registers(self) -> int:
        if self.data_type == ModbusTypes.BOOL:
            return 1
        if self.data_type in [ModbusTypes.INT32, ModbusTypes.FLOAT32]:
            return 2
        return 0

    def to_dict(self, as_primitive: bool = False) -> dict:
        d = super().to_dict(as_primitive)
        d.update({
            "address": self.address,
            "direction": self.direction,
            "field_device": self.field_device,
            "data_type": self.data_type.name if as_primitive else self.data_type,
            "unit_id": self.unit_id
        })
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.address = d["address"]
        self.direction = d["direction"]
        self.field_device = d["field_device"]
        self.data_type = d["data_type"]
        self.unit_id = d["unit_id"]

    def get_protocol_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "address": self.address,
            "direction": self.direction,
            "field_id": self.field_device.name,
            "type_id": self.data_type.value,
            "unit_id": self.unit_id,
            "width": self.width_registers
        }

