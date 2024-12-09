import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.layers.network.configuration.data_point import DataPoint

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel


@dataclass(kw_only=True)
class MMSDataAttribute(DataPoint):
    def get_mms_path(self) -> str:
        return f"{self.get_mms_data_object().get_mms_path()}.{self.name}"

    def __hash__(self):
        return hash(self.uid)

    def get_mms_model(self) -> 'MMSModel':
        return self.get_mms_logical_device().get_mms_model()

    def get_mms_logical_device(self) -> 'MMSLogicalDevice':
        return self.get_mms_logical_node().get_mms_logical_device()

    def get_mms_logical_node(self) -> 'MMSLogicalNode':
        return self.get_mms_data_object().get_mms_logical_node()

    def get_mms_data_object(self) -> 'MMSDataObject':
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject
        data_objects = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSDataObject})
        if len(data_objects) == 0:
            raise ModelNodeNotFoundException(f"Cannot find MMSDataObject for MMSDataAttribute {self.name}")
        if len(data_objects) > 1:
            warnings.warn(f"Multiple data objects associated with MMSDataAttribute {self.name}")
        return data_objects[0]
