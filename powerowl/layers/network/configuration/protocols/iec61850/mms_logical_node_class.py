import enum
import warnings
from typing import List, TYPE_CHECKING

from powerowl.layers.network.configuration.protocols.iec61850.mms_logical_node_group import MmsLogicalNodeGroup
from powerowl.layers.powergrid.elements.enums.line_type import LineType
from powerowl.layers.powergrid.elements.enums.static_generator_type import StaticGeneratorType
from powerowl.layers.powergrid.elements.enums.switch_type import SwitchType

if TYPE_CHECKING:
    from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
    from powerowl.layers.powergrid.elements import GridElement


class MmsLogicalNodeClass(str, enum.Enum):
    # According to DIN EN 61850-7-4
    SWITCH_GEAR = "SWI"
    CIRCUIT_BREAKER = "CBR"
    SYNCHRONIZER = "SYN"
    COUNTER = "CNT"
    OUTPUT_CURVE = "CSD"
    GENERIC_FILTER = "FIL"
    GENERIC_IO = "GIO"
    OUTPUT_LIMIT = "LIM"
    PID_REGULATOR = "PID"
    RAMP_FUNCTION = "RMP"
    SCHEDULE_CONTROLLER = "SCH"
    SETPOINT_CONTROL_FUNCTION = "SPT"
    SIMPLE_MEASUREMENT = "MXN"

    OVERHEAD_LINE = "LIN"
    GENERATOR = "GEN"
    MOTOR = "MOT"
    REACTOR = "REA"
    RESISTOR = "RES"
    HIGH_VOLTAGE_CABLE = "CAB"
    BATTERY = "BAT"
    CAPACITOR_BANK = "CAP"
    POWER_TRANSFORMER = "PTR"
    POWER_SHUNT = "PSH"
    TAP_CHANGER = "LTC"
    LOAD_BREAKER = "SWI"
    LINE_SWITCH = "SWI"
    EARTHING_SWITCH = "SWI"
    FUSE = "FUS"

    @staticmethod
    def get_logical_node_name_from_asset_and_providers(asset: 'GridElement', providers: List['ProviderInfo'], force_generic: bool) -> str:
        has_monitoring = any([provider.domain == "source" for provider in providers])
        has_control = any([provider.domain == "target" for provider in providers])

        if not force_generic:
            from powerowl.layers.powergrid.elements import Switch, Transformer, StaticGenerator, Line, DcLine
            if isinstance(asset, Switch):
                node_class = MmsLogicalNodeClass.SWITCH_GEAR
                if asset.get_property_value("type") == SwitchType.CIRCUIT_BREAKER:
                    node_class = MmsLogicalNodeClass.CIRCUIT_BREAKER
                if has_control:
                    return f"{MmsLogicalNodeGroup.SUPERVISORY_CONTROL.value}{node_class.value}"
                return f"{MmsLogicalNodeGroup.SUPERVISION_AND_MONITORING.value}{node_class.value}"

            # TODO: Transformers
            if isinstance(asset, Transformer):
                # warnings.warn(f"MMS Node Handling for transformer not yet implemented - falling back to generic device")
                return f"{MmsLogicalNodeGroup.POWER_TRANSFORMER_AND_RELATED_FUNCTIONS.value}{MmsLogicalNodeClass.POWER_TRANSFORMER.value}"

            if isinstance(asset, (Line, DcLine)):
                if has_control:
                    warnings.warn(f"A line should not have control providers")
                node_class = MmsLogicalNodeClass.OVERHEAD_LINE
                if asset.get_property_value("type") == LineType.UNDERGROUND_CABLE:
                    node_class = MmsLogicalNodeClass.HIGH_VOLTAGE_CABLE
                return f"{MmsLogicalNodeGroup.FURTHER_POWER_SYSTEM_EQUIPMENT.value}{node_class.value}"

            # TODO: Generators: Wind
            # TODO: Generators: Distributed
            # TODO: Generators: Hydro
            if isinstance(asset, StaticGenerator):
                generator_type: StaticGeneratorType = asset.get_property_value("generator_type")
                if generator_type.get_clear_type() == StaticGeneratorType.WIND:
                    warnings.warn(f"MMS Node Handling for Wind Turbines not yet implemented - falling back to generic device")
                if generator_type.get_clear_type() == StaticGeneratorType.HYDRO:
                    warnings.warn(f"MMS Node Handling for Hydro Generators not yet implemented - falling back to generic device")

        # Default: Generic model
        if has_control:
            return f"{MmsLogicalNodeGroup.GENERIC_FUNCTION_REFERENCES.value}{MmsLogicalNodeClass.GENERIC_IO.value}"
        return f"{MmsLogicalNodeGroup.METERING_AND_MEASUREMENT.value}{MmsLogicalNodeClass.SIMPLE_MEASUREMENT.value}"
