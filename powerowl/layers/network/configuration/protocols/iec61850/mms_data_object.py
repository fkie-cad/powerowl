import warnings
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.graph.model_node import ModelNode

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel


@dataclass(kw_only=True)
class MMSDataObject(ModelNode):
    def get_mms_path(self) -> str:
        return f"{self.get_mms_logical_node().get_mms_path()}.{self.name}"

    def __hash__(self):
        return hash(self.uid)

    def get_mms_model(self) -> 'MMSModel':
        return self.get_mms_logical_device().get_mms_model()

    def get_mms_logical_device(self) -> 'MMSLogicalDevice':
        return self.get_mms_logical_node().get_mms_logical_device()

    def get_mms_logical_node(self) -> 'MMSLogicalNode':
        from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
        logical_nodes = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSLogicalNode})
        if len(logical_nodes) == 0:
            raise ModelNodeNotFoundException(f"Cannot find MMSLogicalNode for MMSDataObject {self.name}")
        if len(logical_nodes) > 1:
            warnings.warn(f"Multiple logical devices associated with MMSDataObject {self.name}")
        return logical_nodes[0]

    def get_mms_data_attributes(self) -> List['MMSDataAttribute']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
        data_attributes = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSDataAttribute})
        return data_attributes
