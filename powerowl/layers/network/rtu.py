import dataclasses
from functools import cmp_to_key
from typing import List

from powerowl.exceptions import DerivationError
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.data_point_type import DataPointType
from powerowl.layers.network.configuration.protocols.iec104_info import IEC104Info
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_set import MMSDataSet
from powerowl.layers.network.configuration.protocols.iec61850.mms_report_control_block import MMSReportControlBlock
from powerowl.layers.network.configuration.protocols.iec61850mms_info import IEC61850MMSInfo
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName
from powerowl.layers.network.field_device import FieldDevice
from powerowl.layers.network.host import Host


@dataclasses.dataclass(eq=False, kw_only=True)
class RTU(FieldDevice):
    def __post_init__(self):
        super().__post_init__()
        self._data_points: List[DataPoint] = []
        self._ioa_group_size = 1000
        self._ioa_step = 10
        self._start_ioa = 5000
        self._group = 0
        self._ioa: int = 10000
        self._base_coa: int = 100
        self._rpc_id: int = 0

    @property
    def coa(self):
        return self._base_coa + self.typed_id

    def get_data_points(self, sort: bool = True) -> List[DataPoint]:
        def compare_data_points(a: DataPoint, b: DataPoint) -> int:
            # Sort by data point type
            if a.get_data_point_type() != b.get_data_point_type():
                if a.get_data_point_type() == DataPointType.DATA_POINT:
                    return -1
                elif b.get_data_point_type() == DataPointType.DATA_POINT:
                    return 1
                else:
                    if a.get_data_point_type().name < b.get_data_point_type().name:
                        return -1
                    return 1
            return 0

        if sort:
            return sorted(self._data_points, key=cmp_to_key(compare_data_points))
        return self._data_points

    def next_device_group(self):
        current_ioa = self._ioa
        next_ioa = self._ioa
        while next_ioa <= current_ioa:
            self._group += 1
            next_ioa = self._start_ioa + self._group * self._ioa_group_size
        self._ioa = next_ioa

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
        protocol.attribute_identifier = f"{str(self.next_ioa()).rjust(6, '0')}"
        self._data_points.append(data_point)

    def add_mms_report_control_block(self, report_control_block: MMSReportControlBlock):
        report_control_block.rpc_id = f"{self.coa}rcb{self.next_rpc_id()}"
        self._data_points.append(report_control_block)

    def add_mms_data_set(self, data_set: MMSDataSet):
        if data_set.data_set_id is None:
            data_set.data_set_id = f"{self.coa}ds{self.next_rpc_id()}"
        self._data_points.append(data_set)

    def next_ioa(self) -> int:
        self._ioa += self._ioa_step
        return self._ioa

    def next_rpc_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id
