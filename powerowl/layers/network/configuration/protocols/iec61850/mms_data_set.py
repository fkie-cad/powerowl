import typing
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.data_point_type import DataPointType

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.rtu import RTU


@dataclass(kw_only=True)
class MMSDataSet(DataPoint):
    data_set_number: ClassVar[int] = 0
    data_set_id: typing.Optional[str] = None

    def __post_init__(self):
        self._data_set_number = MMSDataSet.data_set_number
        MMSDataSet.data_set_number += 1
        super().__post_init__()

    def __hash__(self):
        return hash(self.uid)

    @property
    def data_point_id(self):
        if self.data_set_id is None:
            return f"ds{self._data_set_number}"
        return self.data_set_id

    def get_mms_model(self) -> 'MMSModel':
        return self.get_mms_logical_device().get_mms_model()

    def get_mms_logical_device(self) -> 'MMSLogicalDevice':
        return self.get_mms_logical_node().get_mms_logical_device()

    def get_mms_logical_node(self) -> 'MMSLogicalNode':
        from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
        nodes = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSLogicalNode})
        if len(nodes) == 0:
            raise ModelNodeNotFoundException(f"Cannot find MMSLogicalNode for MMSDataSet {self.name}")
        if len(nodes) > 1:
            warnings.warn(f"Multiple logical nodes associated with MMSDataSet {self.name}")
        return typing.cast(MMSLogicalNode, nodes[0])

    def get_data_point_type(self) -> DataPointType:
        return DataPointType.DATA_SET_61850

    def get_server(self) -> 'RTU':
        from powerowl.layers.network.rtu import RTU
        rtus = self.mlg.get_inter_layer_neighbors(self, neighbor_instance_filter={RTU})
        if len(rtus) == 0:
            raise ModelNodeNotFoundException(f"Cannot find RTU for MMSDataSet {self.name}")
        if len(rtus) > 1:
            warnings.warn(f"Multiple RTUs associated with MMSDataSet {self.name}")
        return typing.cast(RTU, rtus[0])

    def to_data_point_dict(self, as_primitive: bool = False) -> dict:
        return {
            "identifier": self.data_point_id,
            "type": self.get_data_point_type().value,
            "data_set_data": {
                "data_set_identifier": self.data_point_id,
                "server": self.get_server().coa,
                "model": self.get_mms_model().name,
                "logical_device": self.get_mms_logical_device().name,
                "logical_node": self.get_mms_logical_node().name
            }
        }
