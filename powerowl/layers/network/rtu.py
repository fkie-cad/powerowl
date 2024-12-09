import dataclasses
import warnings
from typing import List, Optional, Type

from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
from powerowl.layers.network.configuration.protocols.iec61850mms_info import IEC61850MMSInfo
from powerowl.layers.network.host import Host
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.protocols.iec104_info import IEC104Info
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName
from powerowl.exceptions import DerivationError


@dataclasses.dataclass(eq=False, kw_only=True)
class RTU(Host):
    def __post_init__(self):
        super().__post_init__()
        self._data_points: List[DataPoint] = []
        self._ioa_group_size = 1000
        self._ioa_step = 10
        self._start_ioa = 5000
        self._group = 0
        self._ioa: int = 10000
        self._base_coa: int = 100

    @property
    def coa(self):
        return self._base_coa + self.typed_id

    def get_data_points(self) -> List[DataPoint]:
        return self._data_points

    def next_device_group(self):
        self._ioa = self._start_ioa + self._group * self._ioa_group_size

    def add_104_data_point(self, data_point: DataPoint):
        if not isinstance(data_point.protocol, IEC104Info):
            raise DerivationError(f"Expecting {ProtocolName.IEC104}, got {data_point.protocol.name}")

        protocol = data_point.protocol
        protocol.coa = self.coa
        protocol.ioa = self.next_ioa()
        self._data_points.append(data_point)

    def add_mms_data_point(self, data_point: MMSDataAttribute):
        if not isinstance(data_point.protocol, IEC61850MMSInfo):
            raise DerivationError(f"Expecting {ProtocolName.IEC61850_MMS}, got {data_point.protocol.name}")
        protocol = data_point.protocol
        protocol.attribute_identifier = str(self.next_ioa()).rjust(6, "0")
        self._data_points.append(data_point)

    def next_ioa(self) -> int:
        self._ioa += self._ioa_step
        return self._ioa
