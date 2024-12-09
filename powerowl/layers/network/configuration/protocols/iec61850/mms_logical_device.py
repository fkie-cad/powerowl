import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.graph.model_node import ModelNode

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject


@dataclass(kw_only=True)
class MMSLogicalDevice(ModelNode):
    def get_mms_path(self) -> str:
        return f"{self.get_mms_model().get_mms_path()}{self.name}"

    def __hash__(self):
        return hash(self.uid)

    def get_mms_model(self) -> 'MMSModel':
        from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
        models = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter=[MMSModel])
        if len(models) == 0:
            raise ModelNodeNotFoundException(f"Cannot find MMSModel for MMSLogicalNode {self.name}")
        if len(models) > 1:
            warnings.warn(f"Multiple models associated with logical node {self.name}")
        return models[0]

    def get_mms_logical_nodes(self) -> List['MMSLogicalNode']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
        nodes = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSLogicalNode})
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
