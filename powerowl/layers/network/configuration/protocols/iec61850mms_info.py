from typing import Type, TYPE_CHECKING, List, Optional

from powerowl.layers.network.configuration.data_point_value import DataPointValue
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
from powerowl.layers.network.configuration.protocols.iec61850.mms_functional_constraints import MMSFunctionalConstraints
from powerowl.layers.network.configuration.protocols.iec61850.mms_report_control_block import MMSReportControlBlock
from powerowl.layers.network.configuration.protocols.iec61850.mms_trigger_options import MMSTriggerOptions
from powerowl.layers.network.configuration.protocols.protocol_info import ProtocolInfo
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName
from powerowl.layers.powergrid.values.grid_value_type import Step


if TYPE_CHECKING:
    from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
    from powerowl.layers.network.configuration.protocols.iec61850.mms_model import MMSModel
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_device import MMSLogicalDevice
    from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node import MMSLogicalNode


class IEC61850MMSInfo(ProtocolInfo):
    def __init__(self):
        super().__init__(ProtocolName.IEC61850_MMS)
        self.mms_attribute: Optional[MMSDataAttribute] = None
        self.attribute_identifier: str = "NONE"

        self.server: int = -1
        self.type: str = "NONE"
        self.network_node: Optional = None
        self.functional_constraints: MMSFunctionalConstraints = MMSFunctionalConstraints.NONE
        self.trigger_options: List[MMSTriggerOptions] = []
        self.report_control_blocks: List[MMSReportControlBlock] = []
        self._direction: str = "monitoring"

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction: str):
        direction = direction.lower()
        if direction not in ["monitoring", "control"]:
            raise ValueError("Invalid direction. Must be either 'control' or 'monitoring'")
        self._direction = direction

    def to_dict(self, as_primitive: bool = False) -> dict:
        d = super().to_dict(as_primitive)
        d.update(self.get_protocol_data_dict(as_primitive))
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.attribute_identifier = d["attribute_identifier"]
        self.server = d["server"]
        self.type = d["type"]
        self.functional_constraints = d["functional_constraints"]
        # TODO
        self.trigger_options = []
        # TODO
        self.report_control_blocks = []

    def get_protocol_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "attribute_identifier": self.attribute_identifier,
            "model": self.mms_attribute.get_mms_model().name,
            "logical_device": self.mms_attribute.get_mms_logical_device().name,
            "logical_node": self.mms_attribute.get_mms_logical_node().name,
            "data_object": self.mms_attribute.get_mms_data_object().name,
            "mms_path": self.mms_attribute.get_mms_path(),

            "server": self.server,
            "direction": self.direction,
            "type": self.type,
            "functional_constraints": self.functional_constraints.value,
            "trigger_options": [option.value for option in self.trigger_options],
            # TODO:
            "report_control_blocks": [],
        }

    def generate_data_point_id(self):
        if self.mms_attribute is None:
            raise ValueError("MMSDataAttribute is not set")
        if self.network_node is None:
            raise ValueError("NetworkNode is not set")
        return f"{self.name.name}-{self.network_node.id}-{self.mms_attribute.get_mms_path()}"

    def set_type_by_provider(self, provider: 'ProviderInfo') -> bool:
        from powerowl.layers.network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo
        from powerowl.layers.network.configuration.providers.protection_event_provider_info import ProtectionEventProviderInfo

        if isinstance(provider, PowerGridProviderInfo):
            return self.set_type_by_value_type(provider.attribute_type)
        elif isinstance(provider, ProtectionEventProviderInfo):
            # Protection Event: TODO
            self.type = "PROTECTION"
            return True
        return False

    def set_type_by_value_type(self, data_type: Type[DataPointValue]) -> bool:
        if data_type == float:
            self.type = "FLOAT32"

        if data_type == int:
            self.type = "INT32"

        if data_type == bool:
            self.type = "BOOLEAN"

        if data_type == Step:
            self.type = "INT32"
        return False

    def derive_functional_constraints(self) -> bool:
        if self.direction == "monitoring":
            self.functional_constraints = MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX
        else:
            if self.type == "BOOLEAN":
                self.functional_constraints = MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO
            else:
                self.functional_constraints = MMSFunctionalConstraints.PROCESS_COMMAND_ANALOG_SP
        return True

    def derive_trigger_options(self) -> bool:
        # TODO: Update trigger options
        if self.direction == "control":
            # pass
            return True

        if self.type in [17, 38]:
            # Protection event
            # self.trigger_options.append()
            return True

        if self.type in [9, 11, 13]:
            # self.trigger_options.append(MMSTriggerOptions.DATA_CHANGED)
            return True

        self.trigger_options.append(MMSTriggerOptions.DATA_CHANGED)
        return True
