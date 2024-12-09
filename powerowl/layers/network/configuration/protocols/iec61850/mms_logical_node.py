import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.graph.model_node import ModelNode

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute


@dataclass(kw_only=True)
class MMSLogicalNode(ModelNode):
    def get_mms_path(self) -> str:
        return f"{self.get_mms_logical_device().get_mms_path()}/{self.name}"

    def __hash__(self):
        return hash(self.uid)

    def get_mms_model(self) -> 'MMSModel':
        return self.get_mms_logical_device().get_mms_model()

    def get_mms_logical_device(self) -> 'MMSLogicalDevice':
        from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
        devices = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSLogicalDevice})
        if len(devices) == 0:
            raise ModelNodeNotFoundException(f"Cannot find MMSLogicalDevice for MMSLogicalNode {self.name}")
        if len(devices) > 1:
            warnings.warn(f"Multiple logical devices associated with MMSLogicalNode {self.name}")
        return devices[0]

    def get_mms_data_objects(self) -> List['MMSDataObject']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
        data_objects = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSDataObject})
        return data_objects

    def get_mms_data_attributes(self) -> List['MMSDataAttribute']:
        data_attributes = []
        for data_object in self.get_mms_data_objects():
            data_attributes.extend(data_object.get_mms_data_attributes())
        return data_attributes
