from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from powerowl.graph.model_node import ModelNode

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute


@dataclass(kw_only=True)
class MMSModel(ModelNode):
    def get_mms_path(self) -> str:
        return f"{self.name}"

    def __hash__(self):
        return hash(self.uid)

    def get_mms_logical_devices(self) -> List['MMSLogicalDevice']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
        devices = self.mlg.get_inter_layer_neighbors(self, neighbor_instance_filter={MMSLogicalDevice})
        return devices

    def get_mms_logical_nodes(self) -> List['MMSLogicalNode']:
        nodes = []
        for device in self.get_mms_logical_devices():
            nodes.extend(device.get_mms_logical_nodes())
        return nodes

    def get_mms_data_objects(self) -> List['MMSDataObject']:
        data_objects = []
        for logical_node in self.get_mms_logical_nodes():
            data_objects.extend(logical_node.get_mms_data_objects())
        return data_objects

    def get_mms_data_attributes(self) -> List['MMSDataAttribute']:
        data_attributes = []
        for data_object in self.get_mms_data_objects():
            data_attributes.extend(data_object.get_mms_data_attributes())
        return data_attributes
