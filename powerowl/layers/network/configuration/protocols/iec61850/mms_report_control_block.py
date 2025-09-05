import typing
import warnings
from dataclasses import dataclass, field
from typing import ClassVar, TYPE_CHECKING, List, Optional

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.data_point_type import DataPointType
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_set import MMSDataSet
from powerowl.layers.network.configuration.protocols.iec61850.mms_report_inclusion_options import MMSReportInclusionOptions
from powerowl.layers.network.configuration.protocols.iec61850.mms_trigger_options import MMSTriggerOptions

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_set import MMSDataSet
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice


@dataclass(kw_only=True)
class MMSReportControlBlock(DataPoint):
    report_control_block_number: ClassVar[int] = 0
    trigger_options: List[MMSTriggerOptions] = field(default_factory=list)
    inclusion_options: List[MMSReportInclusionOptions] = field(default_factory=list)
    buffering_time_ms: int = 50
    integrity_period_ms: int = 10000
    configuration_revision: int = 1
    rpc_id: Optional[str] = None

    def __post_init__(self):
        self._rcb_number = MMSReportControlBlock.report_control_block_number
        MMSReportControlBlock.report_control_block_number += 1
        super().__post_init__()

    def __hash__(self):
        return hash(self.uid)

    @property
    def data_point_id(self):
        if self.rpc_id is None:
            return f"rcb{self._rcb_number}"
        return self.rpc_id

    def get_data_set(self, allow_none: bool = False) -> Optional['MMSDataSet']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_set import MMSDataSet
        data_sets = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSDataSet})
        if len(data_sets) == 0:
            if allow_none:
                return None
            raise ModelNodeNotFoundException(f"Cannot find MMSDataSet for MMSReportControlBlock {self.name}")
        if len(data_sets) > 1:
            warnings.warn(f"Multiple MMSDataSets found for MMSReportControlBlock {self.name}")
        return typing.cast(MMSDataSet, data_sets[0])

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
        return DataPointType.REPORT_CONTROL_BLOCK_61850

    def get_server(self) -> int:
        return 0

    def to_data_point_dict(self, as_primitive: bool = False) -> dict:
        data_set_reference = None
        data_set = self.get_data_set(allow_none=True)
        if data_set is not None:
            data_set_reference = f"$ref${data_set.get_data_point_type().value}${data_set.data_point_id}"
        return {
            "identifier": self.data_point_id,
            "type": self.get_data_point_type().value,
            "rcb_data": {
                "rcb_identifier": self.data_point_id,
                "server": self.get_server(),
                "model": self.get_mms_model().name,
                "logical_device": self.get_mms_logical_device().name,
                "logical_node": self.get_mms_logical_node().name,
                "is_buffered": False,
                "data_set": data_set_reference,
                "trigger_options": [trigger_option.value for trigger_option in self.trigger_options],
                "inclusion_options": [inclusion_option.value for inclusion_option in self.inclusion_options],
                "buffering_time": self.buffering_time_ms,
                "integrity_period": self.integrity_period_ms,
                "configuration_revision": self.configuration_revision
            }
        }

