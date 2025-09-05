import dataclasses
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from powerowl.layers.network.configuration.data_point import DataPoint
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute_name import MmsDataAttributeName
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute_type import MmsDataAttributeType
from powerowl.layers.network.configuration.protocols.iec61850.mms_functional_constraints import MMSFunctionalConstraints
from powerowl.layers.network.configuration.protocols.iec61850.mms_trigger_options import MMSTriggerOptions

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
    from powerowl.layers.network.configuration.protocols.iec61850.mms_report_control_block import MMSReportControlBlock
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_set import MMSDataSet
    from powerowl.layers.network.rtu import RTU


@dataclass(kw_only=True)
class MMSDataAttribute(DataPoint):
    data_attribute_type: MmsDataAttributeType = MmsDataAttributeType.UNKNOWN_TYPE
    functional_constraint: MMSFunctionalConstraints = MMSFunctionalConstraints.NONE
    trigger_options: List[MMSTriggerOptions] = dataclasses.field(default_factory=list)

    children: List['MMSDataAttribute'] = dataclasses.field(default_factory=list)
    parent: Optional[typing.Union['MMSDataObject', 'MMSDataAttribute']] = None

    initial_value: Optional[typing.Any] = None
    # Primary attributes are those that should appear in the VCC, for instance
    is_primary_attribute: bool = False

    def __post_init__(self):
        if isinstance(self.name, MmsDataAttributeName):
            self.name = self.name.value
        self.value = self.initial_value
        super().__post_init__()

    def get_mms_path(self) -> str:
        return f"{self.parent.get_mms_path()}.{self.name}"
        # return f"{self.get_mms_data_object().get_mms_path()}.{self.name}"

    def get_parent_data_attributes(self) -> List['MMSDataAttribute']:
        if isinstance(self.parent, MMSDataAttribute):
            return self.parent.get_parent_data_attributes() + [self.parent]
        return []

    def get_parent_data_objects(self):
        return self.get_mms_data_object().get_parent_data_objects() + [self.get_mms_data_object()]

    def __hash__(self):
        return hash(self.uid)

    def get_mms_model(self) -> 'MMSModel':
        return self.get_mms_logical_device().get_mms_model()

    def get_mms_logical_device(self) -> 'MMSLogicalDevice':
        return self.get_mms_logical_node().get_mms_logical_device()

    def get_mms_logical_node(self) -> 'MMSLogicalNode':
        return self.get_mms_data_object().get_mms_logical_node()

    def get_mms_data_attributes(self) -> List['MMSDataAttribute']:
        attributes = []
        for child in self.children:
            if isinstance(child, MMSDataAttribute):
                attributes.append(child)
                attributes.extend(child.get_mms_data_attributes())
        return attributes

    def get_mms_data_object(self) -> 'MMSDataObject':
        if self.parent is None:
            raise ValueError(f"No parent for MMSDataAttribute {self.name}")
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
        if isinstance(self.parent, MMSDataObject):
            return self.parent
        return self.parent.get_mms_data_object()

    def get_report_control_blocks(self) -> List['MMSReportControlBlock']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_report_control_block import MMSReportControlBlock
        report_control_blocks = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSReportControlBlock})
        casted_blocks = [typing.cast(MMSReportControlBlock, report_control_block) for report_control_block in report_control_blocks]
        return casted_blocks

    def get_data_sets(self) -> List['MMSDataSet']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_set import MMSDataSet
        data_sets = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSDataSet})
        casted_data_sets = [typing.cast(MMSDataSet, data_set) for data_set in data_sets]
        return casted_data_sets

    def get_server(self) -> 'RTU':
        return self.get_mms_logical_device().get_ied_node()
        """
        from powerowl.layers.network.rtu import RTU
        rtus = self.mlg.get_inter_layer_neighbors(self, neighbor_instance_filter={RTU})
        if len(rtus) == 0:
            raise ModelNodeNotFoundException(f"Cannot find RTU for MMSDataSet {self.name}")
        if len(rtus) > 1:
            warnings.warn(f"Multiple RTUs associated with MMSDataSet {self.name}")
        return typing.cast(RTU, rtus[0])
        """

    def add_child(self, child: 'MMSDataAttribute'):
        self.children.append(child)
        child.parent = self

    def remove_child(self, child: 'MMSDataAttribute'):
        if child in self.children:
            self.children.remove(child)
            child.parent = None
