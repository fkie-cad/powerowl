import dataclasses
import warnings
from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Union, Optional
from zipfile import compressor_names

from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.graph.enums import EdgeType
from powerowl.graph.model_node import ModelNode
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute_name import MmsDataAttributeName
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute_type import MmsDataAttributeType
from powerowl.layers.network.configuration.protocols.iec61850mms_info import IEC61850MMSInfo
from powerowl.layers.network.network_node import NetworkNode

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel


@dataclass(kw_only=True)
class MMSDataObject(ModelNode):
    children: List[Union['MMSDataObject', 'MMSDataAttribute']] = dataclasses.field(default_factory=list)
    parent: Optional[Union['MMSDataObject', 'MMSLogicalNode']] = None

    def __post_init__(self):
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object_names import MmsDataObjectNames
        if isinstance(self.name, MmsDataObjectNames):
            self.name = self.name.value
        super().__post_init__()

    def get_mms_path(self) -> str:
        return f"{self.parent.get_mms_path()}.{self.name}"

    def __hash__(self):
        return hash(self.uid)

    def get_ied_node(self) -> 'NetworkNode':
        return self.get_mms_logical_device().get_ied_node()

    def get_mms_model(self) -> 'MMSModel':
        return self.get_mms_logical_device().get_mms_model()

    def get_mms_logical_device(self) -> 'MMSLogicalDevice':
        return self.get_mms_logical_node().get_mms_logical_device()

    def get_mms_logical_node(self) -> 'MMSLogicalNode':
        from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode
        if self.parent is not None:
            if isinstance(self.parent, MMSLogicalNode):
                return self.parent
            return self.parent.get_mms_logical_node()

        logical_nodes = self.mlg.get_intra_layer_neighbors(self, neighbor_instance_filter={MMSLogicalNode})
        if len(logical_nodes) == 0:
            raise ModelNodeNotFoundException(f"Cannot find MMSLogicalNode for MMSDataObject {self.name}")
        if len(logical_nodes) > 1:
            warnings.warn(f"Multiple logical devices associated with MMSDataObject {self.name}")
        return logical_nodes[0]

    def get_mms_data_attributes(self) -> List['MMSDataAttribute']:
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
        attributes = []
        for child in self.children:
            if isinstance(child, MMSDataAttribute):
                attributes.append(child)
                attributes.extend(child.get_mms_data_attributes())
        return attributes

    def get_parent_data_objects(self) -> List['MMSDataObject']:
        if isinstance(self.parent, MMSDataObject):
            return self.parent.get_parent_data_objects() + [self.parent]
        return []

    def add_child(self, child: Union['MMSDataAttribute', 'MMSDataObject']):
        self.children.append(child)
        child.parent = self

    def remove_child(self, child: Union['MMSDataAttribute', 'MMSDataObject']):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def _find_cdc_attributes(self,
                             data_attribute_type: MmsDataAttributeType,
                             required_parent_names: List[MmsDataAttributeName],
                             potential_base_names: List[MmsDataAttributeName],
                             root: Optional[Union['MMSDataObject', 'MMSDataAttribute']] = None,
                             base_name_found: bool = False,
                             base_name_is_leaf: bool = False):
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute

        if root is None:
            root = self
        matching_attributes = []

        required_parent_names_str = [p.value for p in required_parent_names]
        potential_base_names_str = [b.value for b in potential_base_names]

        if len(required_parent_names) > 0:
            if root.name == required_parent_names_str[0]:
                required_parent_names = required_parent_names[1:]

            if root.name in potential_base_names_str:
                base_name_found = True

            # Only child attributes are candidates
            for child in root.children:
                matching_attributes.extend(
                    self._find_cdc_attributes(
                        data_attribute_type,
                        required_parent_names,
                        potential_base_names,
                        child,
                        base_name_found,
                        base_name_is_leaf
                    ))
            return matching_attributes

        if root.name in potential_base_names_str:
            base_name_found = True

        # Target attribute has no further children
        if base_name_found:
            if len(root.children) == 0 and isinstance(root, MMSDataAttribute):
                if root.data_attribute_type == data_attribute_type:
                    if not base_name_is_leaf or root.name in potential_base_names_str:
                        matching_attributes.append(root)
            else:
                for child in root.children:
                    matching_attributes.extend(self._find_cdc_attributes(
                        data_attribute_type,
                        required_parent_names,
                        potential_base_names,
                        child,
                        base_name_found,
                        base_name_is_leaf
                    ))

        return matching_attributes

    def _find_attributes(self,
                         data_attribute_type: MmsDataAttributeType,
                         required_parent_names: List[str],
                         name_component_variants: List[str],
                         must_be_leaf: bool = True) -> List['MMSDataAttribute']:

        matching_attributes = []

        for data_attribute in self.get_mms_data_attributes():
            if must_be_leaf and len(data_attribute.children) > 0:
                continue

            matched_parent_names = set()
            matches_components = False

            if data_attribute.name in name_component_variants:
                matches_components = True

            for parent in data_attribute.get_parent_data_objects() + data_attribute.get_parent_data_attributes():
                if parent.name in name_component_variants:
                    matches_components = True
                if parent.name in required_parent_names:
                    matched_parent_names.add(parent.name)

            if matches_components and len(matched_parent_names) == len(required_parent_names):
                matching_attributes.append(data_attribute)

        return matching_attributes

    def find_cdc_status_attributes(self, data_attribute_type: MmsDataAttributeType) -> List['MMSDataAttribute']:
        potential_base_names = [
            MmsDataAttributeName.STATUS_VALUE,
            MmsDataAttributeName.MEASUREMENT_VALUE,
            MmsDataAttributeName.INSTANTANEOUS_MAGNITUDE,
            MmsDataAttributeName.MAGNITUDE,
            MmsDataAttributeName.POSITION_VALUE
        ]

        return self._find_attributes(
            data_attribute_type=data_attribute_type,
            required_parent_names=[],
            name_component_variants=[p.value for p in potential_base_names],
            must_be_leaf=True
        )

        #return self._find_cdc_attributes(
        #    data_attribute_type=data_attribute_type,
        #    required_parent_names=[],
        #    potential_base_names=potential_base_names,
        #    root=self
        #)

    def find_cdc_control_model_attributes(self) -> Optional['MMSDataAttribute']:
        potential_base_names = [
            MmsDataAttributeName.CONTROL_MODEL
        ]
        attributes = self._find_attributes(
            data_attribute_type=MmsDataAttributeType.ENUMERATED,
            required_parent_names=[],
            name_component_variants=[p.value for p in potential_base_names],
            must_be_leaf=True
        )
        if len(attributes) == 1:
            return attributes[0]
        return None

    def find_cdc_control_attributes(self, data_attribute_type: MmsDataAttributeType) -> List['MMSDataAttribute']:
        required_parent_names = [
            MmsDataAttributeName.OPERATE
        ]
        potential_base_names = [
            MmsDataAttributeName.CONTROL_VALUE
        ]

        return self._find_attributes(
            data_attribute_type=data_attribute_type,
            required_parent_names=[p.value for p in required_parent_names],
            name_component_variants=[p.value for p in potential_base_names],
            must_be_leaf=True
        )

        #return self._find_cdc_attributes(
        #    data_attribute_type=data_attribute_type,
        #    required_parent_names=required_parent_names,
        #    potential_base_names=potential_base_names,
        #    root=self
        #)

    def create_protocol_information(self, root: Optional[Union['MMSDataObject', 'MMSDataAttribute']] = None):
        if root is None:
            root = self
        from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
        if isinstance(root, MMSDataAttribute):
            # Add protocol information
            ied = root.get_server()
            root.is_primary_attribute = len(root.providers) > 0

            try:
                direction = root.functional_constraint.get_direction()
            except ValueError as e:
                warnings.warn(root.get_mms_path())
                raise e

            protocol = IEC61850MMSInfo()
            protocol.is_primary_attribute = root.is_primary_attribute
            protocol.network_node = ied
            protocol.mms_attribute = root
            protocol.direction = direction
            protocol.type = root.data_attribute_type.name
            protocol.functional_constraint = root.functional_constraint
            protocol.trigger_options = root.trigger_options
            root.set_auto_name(False)
            root.protocol = protocol
            ied.add_mms_data_point(root)
            self.mlg.build_edge(ied, root, EdgeType.ASSOCIATED)

        for child in root.children:
            self.create_protocol_information(child)
