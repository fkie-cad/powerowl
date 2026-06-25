import dataclasses
from typing import List

from powerowl.exceptions import DerivationError
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.protocols.modbus_tcp_info import ModbusTCPInfo
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName
from powerowl.layers.network.host import Host


@dataclasses.dataclass(eq=False, kw_only=True)
class FieldDevice(Host):
    def __post_init__(self):
        super().__post_init__()
        self._data_points: List[DataPoint] = []
        self._next_address = {
            "cro": 0,
            "crw": 0,
            "rro": 0,
            "rrw": 0
        }

    def get_data_points(self) -> List[DataPoint]:
        return self._data_points

    def add_modbus_data_point(self, data_point: DataPoint):
        if not isinstance(data_point.protocol, ModbusTCPInfo):
            raise DerivationError(f"Expecting {ProtocolName.MODBUS_TCP}, got {data_point.protocol.name}")
        table = data_point.protocol.table
        if table not in ["cro", "crw", "rro", "rrw"]:
            raise ValueError(f"Invalid table name: {table}. "
                             f"Expecting 'cro' (read-only coil), 'crw' (discrete input coil), "
                             f"'rro' (input register) or 'rrw' (holding register)")
        data_point.protocol.pdu_address = self._next_address[table]
        self._next_address[table] += data_point.protocol.width_registers
        self._data_points.append(data_point)
