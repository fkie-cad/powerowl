from typing import Optional, TYPE_CHECKING, List

from powerowl.layers.network.configuration.protocols.modbus.modbus_function_codes import ModbusFunctionCode
from powerowl.layers.network.configuration.protocols.protocol_info import ProtocolInfo
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName
from powerowl.layers.network.configuration.providers.endian import Endian
from powerowl.layers.network.configuration.providers.modbus_types import ModbusTypes

if TYPE_CHECKING:
    from powerowl.layers.network.field_device import FieldDevice


class ModbusTCPInfo(ProtocolInfo):
    def __init__(self):
        super().__init__(ProtocolName.MODBUS_TCP)
        self._direction = "monitoring"
        self._absolute_address: int = 0
        self.field_device: Optional['FieldDevice'] = None
        self.data_type: ModbusTypes = ModbusTypes.INT32
        self.unit_id: int = 0
        self.endian = Endian.BIG_ENDIAN
        self.polling_interval: int = 0

    def generate_data_point_id(self):
        return f"{self.field_device.uid}.{self.unit_id}.{self.table}.{self.address}"

    def derive_data_type(self):
        if self.direction == "monitoring" and self.data_type == ModbusTypes.STEP:
            self.data_type = ModbusTypes.INT32

    @property
    def table(self) -> str:
        suffix = "ro" if self.direction == "monitoring" else "rw"
        if self.data_type in [ModbusTypes.BOOL, ModbusTypes.STEP]:
            return f"c{suffix}"
        return f"r{suffix}"

    def get_table_name(self):
        return {
            "cro": "discrete input",
            "crw": "coil",
            "rro": "input register",
            "rrw": "holding register",
        }.get(self.table)

    @property
    def address(self):
        """
        The ABSOLUTE address of the register
        """
        return self._absolute_address

    @address.setter
    def address(self, address: int):
        """
        Set the ABSOLUTE address of the register
        """
        self._absolute_address = address

    @property
    def table_offset(self) -> int:
        """
        The table offset depends on the register type and is hard-coded.
        """
        match self.get_table_name():
            case "coil":
                return 0
            case "discrete input":
                return 10000
            case "input register":
                return 30000
            case "holding register":
                return 40000
        raise ValueError("Invalid table name")

    @property
    def pdu_address(self) -> int:
        """
        The address to use in a Modbus PDU to access the register.
        This assumes 0-based addressing
        """
        return self.address - self.table_offset

    @pdu_address.setter
    def pdu_address(self, address: int):
        self._absolute_address = address + self.table_offset

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
        if self.data_type in [ModbusTypes.BOOL, ModbusTypes.STEP]:
            return 1
        if self.data_type in [ModbusTypes.INT32, ModbusTypes.FLOAT32]:
            return 2
        if self.data_type in [ModbusTypes.INT64, ModbusTypes.DOUBLE]:
            return 4
        return 0

    @property
    def supported_function_codes(self) -> List[int]:
        match self.get_table_name():
            case "coil":
                return [ModbusFunctionCode.READ_COIL.value, ModbusFunctionCode.WRITE_COIL.value, ModbusFunctionCode.WRITE_MULTIPLE_COILS.value]
            case "discrete input":
                return [ModbusFunctionCode.READ_DISCRETE_INPUT.value]
            case "input register":
                return [ModbusFunctionCode.READ_INPUT_REGISTER.value]
            case "holding register":
                return [ModbusFunctionCode.READ_HOLDING_REGISTER.value, ModbusFunctionCode.WRITE_HOLDING_REGISTER.value, ModbusFunctionCode.WRITE_HOLDING_REGISTERS.value]
        return []

    def to_dict(self, as_primitive: bool = False) -> dict:
        d = super().to_dict(as_primitive)
        d.update({
            "address": self.address,
            "table_offset": self.table_offset,
            "zero_based_pdu_address": self.pdu_address,
            "direction": self.direction,
            "device_id": self.field_device.uid,
            "data_type": self.data_type.name if as_primitive else self.data_type,
            "unit_id": self.unit_id,
            "polling_interval": self.polling_interval,
            "supported_function_codes": self.supported_function_codes,
        })
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.address = d["address"]
        self.direction = d["direction"]
        self.field_device = d["field_device"]
        self.data_type = d["data_type"]
        self.unit_id = d["unit_id"]
        self.polling_interval = d["polling_interval"]

    def get_protocol_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "address": self.address,
            "table_offset": self.table_offset,
            "zero_based_pdu_address": self.pdu_address,
            "table": self.get_table_name(),
            "direction": self.direction,
            "device_id": self.field_device.uid,
            "type_id": self.data_type.value,
            "unit_id": self.unit_id,
            "width": self.width_registers,
            "endian": self.endian.value if as_primitive else self.endian.name,
            "polling_enabled": self.polling_interval > 0,
            "polling_interval": self.polling_interval,
            "supported_function_codes": self.supported_function_codes,
            "zero_based_address": True
        }

    def derive_polling_interval(self, polling_interval: int = 10):
        self.polling_interval = 0
        if self.data_type.to_native() in [int, float] and self.direction == "monitoring":
            self.polling_interval = polling_interval
