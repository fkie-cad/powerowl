import enum
from typing import Tuple, List, Optional

from powerowl.layers.network.configuration.protocols.iec61850.mms_common_data_class import MmsCommonDataClass
from powerowl.layers.powergrid.elements import GridElement, Switch
from powerowl.layers.powergrid.values.grid_value import GridValue
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext
from powerowl.layers.powergrid.values.grid_value_type import Step
from powerowl.layers.powergrid.values.units.unit import Unit


class MmsDataObjectNames(str, enum.Enum):
    SWITCH_POSITION = "Pos"
    TAP_CHANGER = "LTC"

    CURRENT = "A"
    AMPERE = "Amp"
    VOLTAGE = "V"
    VOLTAGE_SIDE_1 = "V1"
    VOLTAGE_SIDE_2 = "V2"
    POWER_FACTOR_PREFIX = "PF"
    ACTIVE_POWER_INSTANTANEOUS = "Pi"
    ACTIVE_POWER = "W"
    REACTIVE_POWER_INSTANTANEOUS = ""
    REACTIVE_POWER = "VAr"
    APPARENT_POWER = "VA"
    VOLT_AMPERES = "VA"
    LOSS = "Los"
    LOADING = "Lod"
    ANGLE = "Ang"
    PER_UNIT = "Pu"
    RESISTANCE = "Ris"

    @staticmethod
    def get_object_name_from_grid_value(grid_element: GridElement, grid_value: GridValue,
                                        related_options: Optional[List[GridValue]] = None
                                        ) -> Tuple[str, Optional[MmsCommonDataClass], GridValue, Optional[GridValue]]:
        object_name = None
        cdc = None
        status_value = None
        control_value = None

        if related_options is None:
            related_options = []
        related_options = related_options

        if isinstance(grid_element, Switch):
            object_name, cdc, status_value, control_value = MmsDataObjectNames._handle_switch(
                grid_element=grid_element, grid_value=grid_value, related_options=related_options
            )

        if object_name is None:
            object_name = MmsDataObjectNames._get_default_object_name(grid_element=grid_element, grid_value=grid_value)
        if status_value is None:
            status_value = grid_value
        if cdc is None:
            from powerowl.layers.powergrid.elements import Load
            cdc = MmsDataObjectNames._get_fallback_cdc(grid_element=grid_element, grid_value=grid_value, related_options=related_options)

            if grid_value.value_context == GridValueContext.CONFIGURATION:
                control_value = grid_value
                status_value = grid_value
                # Disabled for now: CONFIGURATION status != Measurement (usually?)
                if False and len(related_options) == 1 and related_options[0].value_context == GridValueContext.MEASUREMENT:
                    control_value = grid_value
                    status_value = related_options[0]
            elif grid_value.value_context == GridValueContext.MEASUREMENT:
                status_value = grid_value
                control_value = None

                if len(related_options) == 1 and related_options[0].value_context == GridValueContext.CONFIGURATION:
                    control_value = related_options[0]

        # Fallback: Grid Value attribute name
        return object_name, cdc, status_value, control_value

    @staticmethod
    def _get_default_object_name(grid_element: GridElement, grid_value: GridValue) -> str:
        original_name = grid_value.name
        forbidden_parts = ["target", "is"]
        short_name = "".join([part[0] for part in original_name.split("_") if len(part) > 0 and part not in forbidden_parts])
        object_name = short_name
        if grid_value.value_context in [GridValueContext.MEASUREMENT, GridValueContext.CONFIGURATION]:
            if "loss" in grid_value.name:
                object_name = f"{object_name}{MmsDataObjectNames.LOSS.value}"
            match grid_value.unit:
                case Unit.AMPERE:
                    object_name = f"{object_name}{MmsDataObjectNames.AMPERE.value}"
                case Unit.VOLT:
                    object_name = f"{object_name}{MmsDataObjectNames.VOLTAGE.value}"
                case Unit.WATT:
                    object_name = f"{object_name}{MmsDataObjectNames.ACTIVE_POWER.value}"
                case Unit.VAR:
                    object_name = f"{object_name}{MmsDataObjectNames.REACTIVE_POWER.value}"
                case Unit.DEGREE:
                    object_name = f"{object_name}{MmsDataObjectNames.ANGLE.value}"
                case Unit.PER_UNIT:
                    object_name = f"{object_name}{MmsDataObjectNames.PER_UNIT.value}"
                case Unit.VOLT_AMPERE:
                    object_name = f"{object_name}{MmsDataObjectNames.VOLT_AMPERES.value}"
                case Unit.OHM:
                    object_name = f"{object_name}{MmsDataObjectNames.RESISTANCE.value}"
                case Unit.PERCENT:
                    if "loading" in grid_value.name:
                        object_name = f"{object_name}{MmsDataObjectNames.LOADING.value}"
        return object_name

    @staticmethod
    def _handle_switch(grid_element: Switch, grid_value: GridValue,
                       related_options: Optional[List[GridValue]] = None) -> Tuple[str, str, GridValue, Optional[GridValue]]:
        object_name = None
        cdc = None
        status_value = None
        control_value = None

        # SPC / DPC for status
        if grid_value.value_context == GridValueContext.CONFIGURATION and grid_value.name == "closed":
            object_name = MmsDataObjectNames.SWITCH_POSITION
            cdc = MmsCommonDataClass.CONTROLLABLE_SINGLE_POINT
            control_value = grid_value
            status_value = grid_value

            if len(related_options) == 1:
                if related_options[0] in grid_value.get_related_grid_values():
                    # Related measurement found - use this instead
                    status_value = related_options[0]

        if grid_value.value_context == GridValueContext.MEASUREMENT and grid_value.name == "is_closed":
            object_name = MmsDataObjectNames.SWITCH_POSITION
            cdc = MmsCommonDataClass.SINGLE_POINT_STATUS_INFORMATION
            if len(related_options) == 1:
                if related_options[0] in grid_value.get_related_grid_values():
                    # Control value found - use this instead
                    cdc = MmsCommonDataClass.CONTROLLABLE_SINGLE_POINT
                    status_value = grid_value
                    control_value = related_options[0]

        return object_name, cdc, status_value, control_value

    @staticmethod
    def _get_fallback_cdc(grid_element: GridElement, grid_value: GridValue, related_options: List[GridValue]) -> Optional[MmsCommonDataClass]:
        if grid_value.value_type == float:
            if grid_value.value_context == GridValueContext.CONFIGURATION:
                return MmsCommonDataClass.CONTROLLABLE_ANALOGUE_PROCESS_VALUE
            return MmsCommonDataClass.MEASURED_VALUE

        if grid_value.value_type == int:
            if grid_value.value_context == GridValueContext.CONFIGURATION:
                return MmsCommonDataClass.CONTROLLABLE_INTEGER_STATUS
            return MmsCommonDataClass.INTEGER_STATUS

        if grid_value.value_type == bool:
            if grid_value.value_context == GridValueContext.CONFIGURATION:
                return MmsCommonDataClass.CONTROLLABLE_SINGLE_POINT
            return MmsCommonDataClass.SINGLE_POINT_STATUS_INFORMATION

        if grid_value.value_type == Step:
            if grid_value.value_context == GridValueContext.CONFIGURATION:
                return MmsCommonDataClass.BINARY_CONTROLLED_STEP_POSITION
            return MmsCommonDataClass.INTEGER_STATUS

        return None
