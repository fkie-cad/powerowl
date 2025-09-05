import enum
from inspect import signature
from typing import TYPE_CHECKING, Optional, List, Union

from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute import MMSDataAttribute
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute_name import MmsDataAttributeName
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_attribute_type import MmsDataAttributeType
from powerowl.layers.network.configuration.protocols.iec61850.mms_functional_constraints import MMSFunctionalConstraints
from powerowl.layers.network.configuration.protocols.iec61850.mms_trigger_options import MMSTriggerOptions

#if TYPE_CHECKING:
from powerowl.layers.network.configuration.protocols.iec61850.mms_data_object import MMSDataObject


class MmsCommonDataClass(str, enum.Enum):
    # Status Information
    SINGLE_POINT_STATUS_INFORMATION = "SPS"
    DOUBLE_POINT_STATUS = "DPS"
    INTEGER_STATUS = "ISI"
    ENUMERATED_STATUS = "ENS"
    PROTECTION_ACTIVATION = "ACT"
    DIRECTIONAL_PROTECTION_ACTIVATION = "ACD"
    SECURITY_VIOLATION_COUNTING = "SEC"
    BINARY_COUNTER_READING = "BCR"
    HISTOGRAM = "HST"
    VISIBLE_STRING_STATUS = "VSS"

    # Measurands
    MEASURED_VALUE = "MV"
    COMPLEX_MEASURED_VALUE = "CMV"
    SAMPLED_VALUE = "SAV"
    WYE = "WYE"
    DELTA = "DEL"
    SEQUENCE = "SEQ"
    HARMONIC_VALUE = "HMV"
    HARMONIC_VALUE_FOR_WYE = "HWYE"
    HARMONIC_VALUE_FOR_DELTA = "HDEL"

    # Controls
    CONTROLLABLE_SINGLE_POINT = "SPC"
    CONTROLLABLE_DOUBLE_POINT = "DPC"
    CONTROLLABLE_INTEGER_STATUS = "ISC"
    CONTROLLABLE_ENUMERATED_STATUS = "ENC"
    BINARY_CONTROLLED_STEP_POSITION = "BSC"
    INTEGER_CONTROLLED_STEP_POSITION = "ISC"
    CONTROLLABLE_ANALOGUE_PROCESS_VALUE = "APC"
    BINARY_CONTROLLED_ANALOG_PROCESS_VALUE = "BAC"

    # Status Settings
    SINGLE_POINT_SETTING = "SPG"
    INTEGER_STATUS_SETTING = "ISG"
    ENUMERATED_STATUS_SETTING = "ENG"
    OBJECT_REFERENCE_SETTING = "ORG"
    TIME_SETTING_GROUP = "TSG"
    CURRENCY_SETTING_GROUP = "CUG"
    VISIBLE_STRING_SETTING = "VSG"

    # Analogue Settings
    ANALOGUE_SETTING = "ASG"
    SETTING_CURVE = "CURVE"
    CURVE_SHAPE_SETTING_GROUP = "CUG"

    def create_cdc_data_object(self, cdc: 'MmsCommonDataClass', name: str, data_attribute_type: Optional[MmsDataAttributeType] = None, *,
                               add_description: bool = False, add_cdc_name: bool = False, add_data_name_space: bool = False, add_control_number: bool = False,
                               add_pics_substitution: bool = False, add_blocking: bool = False, add_unit: bool = False, add_frozen_value: bool = False,
                               add_address: bool = False, add_info: bool = False, add_instant_measurement: bool = False, add_range: bool = False,
                               add_unit_multiplier: bool = False, create_phase_a: bool = False, create_phase_b: bool = False, create_phase_c: bool = False,
                               add_phase_neutral: bool = False, add_origin: bool = False, use_integrity_for_mag: bool = False, **kwargs
                               ) -> 'MMSDataObject':
        """
        Given a common data class, a respective data object with child attributes is created according to IEC61850-7.
        """
        options = {}
        sig = signature(self.create_cdc_data_object)
        for parameter in sig.parameters.values():
            if parameter.kind == parameter.KEYWORD_ONLY:
                options[parameter.name] = locals()[parameter.name]
        options.update(kwargs)

        match cdc:
            # STATUS INFORMATION
            case MmsCommonDataClass.SINGLE_POINT_STATUS_INFORMATION:
                return self._create_cdc_sps(name, options)
            case MmsCommonDataClass.DOUBLE_POINT_STATUS:
                return self._create_cdc_dps(name, options)
            case MmsCommonDataClass.INTEGER_STATUS:
                return self._create_cdc_ins(name, options)
            case MmsCommonDataClass.ENUMERATED_STATUS:
                return self._create_cdc_ens(name, options)

            # MEASURANDS
            case MmsCommonDataClass.MEASURED_VALUE:
                if data_attribute_type is None:
                    raise AttributeError(f"{MmsCommonDataClass.MEASURED_VALUE.name} requires a valid data attribute type")
                return self._create_cdc_mv(name, data_attribute_type=data_attribute_type, options=options)

            # CONTROL
            case MmsCommonDataClass.CONTROLLABLE_SINGLE_POINT:
                return self._create_cdc_spc(name, options)
            case MmsCommonDataClass.CONTROLLABLE_DOUBLE_POINT:
                return self._create_cdc_dpc(name, options)
            case MmsCommonDataClass.CONTROLLABLE_INTEGER_STATUS:
                return self._create_cdc_isc(name, options)
            case MmsCommonDataClass.INTEGER_CONTROLLED_STEP_POSITION:
                return self._create_cdc_isc(name, options)
            case MmsCommonDataClass.BINARY_CONTROLLED_STEP_POSITION:
                return self._create_cdc_bsc(name, options)
            case MmsCommonDataClass.CONTROLLABLE_ANALOGUE_PROCESS_VALUE:
                if data_attribute_type is None:
                    raise AttributeError(f"{MmsCommonDataClass.CONTROLLABLE_ANALOGUE_PROCESS_VALUE.name} requires a valid data attribute type")
                return self._create_cdc_apc(name, options, data_attribute_type=data_attribute_type)

            # ANALOG STATUS
            # ANALOG SETTINGS

            case _:
                raise NotImplementedError(f"CDC object creation not implemented for {cdc.name} ({cdc.value})")

    """
    STATUS
    """
    def _create_cdc_sps(self, name: str, options: dict) -> 'MMSDataObject':
        data_object = MMSDataObject(name=name)
        self._add_status(data_object, status_type=MmsDataAttributeType.BOOLEAN)
        # TODO: Add optional PicsSubst?
        # TODO: Add optional Block Enabled?
        self._add_standard_options(data_object, options)
        return data_object

    def _create_cdc_dps(self, name: str, options: dict) -> 'MMSDataObject':
        data_object = MMSDataObject(name=name)
        self._add_status(data_object, status_type=MmsDataAttributeType.CODEDENUM)
        # TODO: Add optional PicsSubst?
        # TODO: Add optional Block Enabled?
        self._add_standard_options(data_object, options)
        return data_object

    def _create_cdc_ins(self, name: str, options: dict) -> 'MMSDataObject':
        data_object = MMSDataObject(name=name)
        self._add_status(data_object, status_type=MmsDataAttributeType.INT32)
        self._add_standard_options(data_object, options)
        return data_object

    def _create_cdc_ens(self, name: str, options: dict) -> 'MMSDataObject':
        data_object = MMSDataObject(name=name)
        self._add_status(data_object, status_type=MmsDataAttributeType.ENUMERATED)
        self._add_standard_options(data_object, options)
        return data_object

    """
    MEASURANDS
    """
    def _create_cdc_mv(self, name: str, data_attribute_type: MmsDataAttributeType, options: dict) -> 'MMSDataObject':
        data_object = MMSDataObject(name=name)
        if options.get("add_instant_measurement"):
            self._add_analogue_value(data_object,
                                     name=MmsDataAttributeName.INSTANTANEOUS_MAGNITUDE,
                                     data_attribute_type=data_attribute_type,
                                     functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX,
                                     trigger_options=[MMSTriggerOptions.INTEGRITY])
        trigger_options = [MMSTriggerOptions.DATA_CHANGED, MMSTriggerOptions.DATA_UPDATE]
        if options.get("use_integrity_for_mag", False):
            trigger_options = [MMSTriggerOptions.INTEGRITY]

        self._add_analogue_value(data_object,
                                 name=MmsDataAttributeName.MAGNITUDE,
                                 data_attribute_type=data_attribute_type,
                                 functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX,
                                 trigger_options=trigger_options)

        if options.get("add_range"):
            range_attribute = MMSDataAttribute(name=MmsDataAttributeName.RANGE,
                                               data_attribute_type=MmsDataAttributeType.ENUMERATED,
                                               functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX,
                                               trigger_options=[MMSTriggerOptions.DATA_CHANGED])
            data_object.add_child(range_attribute)
        self._add_standard_options(data_object, options)
        return data_object

    """
    CONTROL
    """
    def _create_cdc_spc(self, name: str, options: dict) -> 'MMSDataObject':
        spc_object = MMSDataObject(name=name)

        self._add_originator_and_control_number(spc_object, options=options)

        self._add_status(spc_object, status_type=MmsDataAttributeType.BOOLEAN)

        # Control value via Oper
        self._add_controls(spc_object, MmsDataAttributeType.BOOLEAN, options)
        self._add_common_control_attributes(spc_object, options)

        self._add_standard_options(spc_object, options)

        return spc_object

    def _create_cdc_dpc(self, name: str, options: dict) -> 'MMSDataObject':
        dpc_object = MMSDataObject(name=name)

        self._add_originator_and_control_number(dpc_object, options=options)

        self._add_status(dpc_object, status_type=MmsDataAttributeType.CODEDENUM)

        # Control value via Oper
        self._add_controls(dpc_object, MmsDataAttributeType.BOOLEAN, options)

        self._add_common_control_attributes(dpc_object, options)
        self._add_standard_options(dpc_object, options)
        return dpc_object

    def _create_cdc_isc(self, name: str, options: dict):
        isc_object = MMSDataObject(name=name)

        self._add_originator_and_control_number(isc_object, options=options)

        # Value with potential transient indicator
        self._add_value_with_transient_indicator(isc_object,
                                                 name=MmsDataAttributeName.VALUE_WITH_TRANSIENT_INDICATOR,
                                                 functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST,
                                                 trigger_options=[MMSTriggerOptions.DATA_CHANGED],
                                                 has_transient=options.get("add_transient_indicator", False))

        self._add_time_and_quality(isc_object, functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST)

        self._add_controls(isc_object, MmsDataAttributeType.INT8, options)

        # TODO: PICS?
        # TODO: blkEna?

        if options.get("add_minimum_value", False):
            isc_object.add_child(MMSDataAttribute(name=MmsDataAttributeName.MINIMUM_VALUE,
                                                  data_attribute_type=MmsDataAttributeType.INT32,
                                                  functional_constraint=MMSFunctionalConstraints.CONFIGURATION_CF))
        if options.get("add_maximum_value", False):
            isc_object.add_child(MMSDataAttribute(name=MmsDataAttributeName.MAXIMUM_VALUE,
                                                  data_attribute_type=MmsDataAttributeType.INT32,
                                                  functional_constraint=MMSFunctionalConstraints.CONFIGURATION_CF))

        self._add_common_control_attributes(isc_object, options)
        self._add_standard_options(isc_object, options)
        return isc_object

    def _create_cdc_bsc(self, name: str, options: dict) -> 'MMSDataObject':
        # BINARY STEP COMMAND
        bsc_object = MMSDataObject(name=name)
        self._add_originator_and_control_number(bsc_object, options=options)
        self._add_value_with_transient_indicator(bsc_object,
                                                 name=MmsDataAttributeName.VALUE_WITH_TRANSIENT_INDICATOR,
                                                 functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST,
                                                 trigger_options=[MMSTriggerOptions.DATA_CHANGED],
                                                 has_transient=options.get("add_transient_indicator", False))
        self._add_time_and_quality(bsc_object, functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST)

        bsc_object.add_child(MMSDataAttribute(name=MmsDataAttributeName.PERSISTENT,
                                              data_attribute_type=MmsDataAttributeType.BOOLEAN,
                                              functional_constraint=MMSFunctionalConstraints.CONFIGURATION_CF,
                                              trigger_options=[MMSTriggerOptions.DATA_CHANGED]))

        self._add_controls(bsc_object, MmsDataAttributeType.CODEDENUM, options)
        self._add_common_control_attributes(bsc_object, options)
        # TODO: PICS?
        # TODO: blkEna?
        self._add_standard_options(bsc_object, options)
        return bsc_object

    def _create_cdc_apc(self, name: str, options: dict, data_attribute_type: MmsDataAttributeType) -> 'MMSDataObject':
        apc_object = MMSDataObject(name=name)
        self._add_control_status_for_analog_control(apc_object, options=options)
        self._add_analogue_value(apc_object,
                                 name=MmsDataAttributeName.MEASUREMENT_VALUE,
                                 data_attribute_type=data_attribute_type,
                                 functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX,
                                 trigger_options=[MMSTriggerOptions.DATA_CHANGED])
        self._add_time_and_quality(apc_object, functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX)
        self._add_common_control_attributes(apc_object, options)

        # TODO PICS SUBST?
        # TODO: blkEna
        self._add_analog_controls(apc_object, options, data_attribute_type=data_attribute_type)
        self._add_standard_options(apc_object, options)

        return apc_object

    """
    HELPERS
    """
    # STATUS & MEASUREMENT
    def _add_status(self, data_object: 'MMSDataObject', status_type: MmsDataAttributeType) -> bool:
        data_attribute = MMSDataAttribute(name=MmsDataAttributeName.STATUS_VALUE,
                                          data_attribute_type=status_type,
                                          functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST,
                                          trigger_options=[MMSTriggerOptions.DATA_UPDATE, MMSTriggerOptions.DATA_CHANGED])
        data_object.add_child(data_attribute)
        self._add_time_and_quality(data_object, MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST)
        return True

    def _add_time_and_quality(self, data_object: 'MMSDataObject', functional_constraint: MMSFunctionalConstraints) -> bool:
        time_attribute = MMSDataAttribute(name=MmsDataAttributeName.TIMESTAMP,
                                          data_attribute_type=MmsDataAttributeType.TIMESTAMP,
                                          functional_constraint=functional_constraint)
        quality_attribute = MMSDataAttribute(name=MmsDataAttributeName.QUALITY,
                                             data_attribute_type=MmsDataAttributeType.QUALITY,
                                             functional_constraint=functional_constraint,
                                             trigger_options=[MMSTriggerOptions.QUALITY_CHANGED])
        data_object.add_child(time_attribute)
        data_object.add_child(quality_attribute)
        return True

    def _add_standard_options(self, data_object: 'MMSDataObject', options: dict) -> bool:
        if options.get("add_description"):
            desc_attribute = MMSDataAttribute(name=MmsDataAttributeName.DESCRIPTION,
                                              data_attribute_type=MmsDataAttributeType.VISIBLE_STRING_255,
                                              functional_constraint=MMSFunctionalConstraints.DESCRIPTION_DC)
            data_object.add_child(desc_attribute)

            unicode_desc_attribute = MMSDataAttribute(
                name=MmsDataAttributeName.DESCRIPTION,
                data_attribute_type=MmsDataAttributeType.UNICODE_STRING_255,
                functional_constraint=MMSFunctionalConstraints.DESCRIPTION_DC
                )
            data_object.add_child(desc_attribute)
            data_object.add_child(unicode_desc_attribute)
        # TODO: Respect further options

    def _add_value_with_transient_indicator(self, parent, name: str, functional_constraint: MMSFunctionalConstraints,
                                            trigger_options: List[MMSTriggerOptions], has_transient: bool):
        # Value with Transient Indicator
        value_with_transient = MMSDataAttribute(
            name=name,
            data_attribute_type=MmsDataAttributeType.CONSTRUCTED,
            functional_constraint=functional_constraint,
            trigger_options=trigger_options
        )
        parent.add_child(value_with_transient)

        # Add position Value
        value_with_transient.add_child(
            MMSDataAttribute(
                name=MmsDataAttributeName.POSITION_VALUE,
                data_attribute_type=MmsDataAttributeType.INT8,
                functional_constraint=functional_constraint,
                trigger_options=trigger_options
            )
        )

        # Add Transient Indicator
        if has_transient:
            value_with_transient.add_child(
                MMSDataAttribute(
                    name=MmsDataAttributeName.TRANSIENT_INDICATOR,
                    data_attribute_type=MmsDataAttributeType.BOOLEAN,
                    functional_constraint=functional_constraint,
                    trigger_options=trigger_options
                )
            )

    def _add_analogue_value(self, parent: Union['MMSDataObject', 'MMSDataAttribute'], name: str, data_attribute_type: MmsDataAttributeType,
                            functional_constraint: MMSFunctionalConstraints, trigger_options: List[MMSTriggerOptions]) -> bool:

        if data_attribute_type in [
            MmsDataAttributeType.INT8,
            MmsDataAttributeType.INT16,
            MmsDataAttributeType.INT32,
            MmsDataAttributeType.INT64,
            MmsDataAttributeType.INT128,
            MmsDataAttributeType.INT8U,
            MmsDataAttributeType.INT16U,
            MmsDataAttributeType.INT32U
        ]:
            is_int = True
        elif data_attribute_type in [
            MmsDataAttributeType.FLOAT32,
            MmsDataAttributeType.FLOAT64,
        ]:
            is_int = False
        else:
            raise ValueError(f"Unsupported data attribute type for analogue value: {data_attribute_type.name}")

        analog_attribute = MMSDataAttribute(name=name,
                                            data_attribute_type=MmsDataAttributeType.CONSTRUCTED,
                                            functional_constraint=functional_constraint,
                                            trigger_options=trigger_options)
        parent.add_child(analog_attribute)

        child_name = "i" if is_int else "f"
        value_attribute = MMSDataAttribute(name=child_name,
                                           data_attribute_type=data_attribute_type,
                                           functional_constraint=functional_constraint,
                                           trigger_options=trigger_options)
        analog_attribute.add_child(value_attribute)
        return True

    # CONTROL
    def _add_controls(self, data_object, data_attribute_type: MmsDataAttributeType, options: dict):
        # TODO: Allow SBO?
        # 1 = DIRECT NORMAL
        control_mode = 1
        control_model = MMSDataAttribute(name=MmsDataAttributeName.CONTROL_MODEL,
                                         data_attribute_type=MmsDataAttributeType.ENUMERATED,
                                         functional_constraint=MMSFunctionalConstraints.CONFIGURATION_CF,
                                         trigger_options=[MMSTriggerOptions.DATA_CHANGED],
                                         initial_value=control_mode)
        data_object.add_child(control_model)
        if control_mode != 0:
            # Add Oper
            self._add_operate(data_object, data_attribute_type=data_attribute_type, time_activated=False, options=options)

    def _add_analog_controls(self, parent, options: dict, data_attribute_type: MmsDataAttributeType):
        # TODO: Allow SBO
        control_mode = 1
        control_model = MMSDataAttribute(name=MmsDataAttributeName.CONTROL_MODEL,
                                         data_attribute_type=MmsDataAttributeType.ENUMERATED,
                                         functional_constraint=MMSFunctionalConstraints.CONFIGURATION_CF,
                                         trigger_options=[MMSTriggerOptions.DATA_CHANGED],
                                         initial_value=control_mode)
        parent.add_child(control_model)
        if control_mode != 0:
            # TODO: Check for SBO?
            # TODO; Check for time activated
            # TODO: Check for SBO_ENHANCED
            # Add Operate
            operate = MMSDataAttribute(name=MmsDataAttributeName.OPERATE,
                                       data_attribute_type=MmsDataAttributeType.CONSTRUCTED,
                                       functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO)
            parent.add_child(operate)
            self._add_analogue_value(
                parent=operate,
                name=MmsDataAttributeName.CONTROL_VALUE,
                functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO,
                data_attribute_type=data_attribute_type,
                trigger_options=[])
            self._add_common_operate_elements(operate, time_activated=False, has_check=True, options=options)

            # TODO: Check for Cancel

    def _add_common_control_attributes(self, data_object, options: dict):
        # TODO
        pass

    def _add_operate(self, data_object, data_attribute_type: MmsDataAttributeType, time_activated: bool, options: dict):
        attribute = MMSDataAttribute(name=MmsDataAttributeName.OPERATE,
                                     data_attribute_type=MmsDataAttributeType.CONSTRUCTED,
                                     functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO)
        data_object.add_child(attribute)
        self._add_generic_operate_elements(data_attribute=attribute,
                                           data_attribute_type=data_attribute_type,
                                           time_activated=time_activated,
                                           has_check=True,
                                           options=options)

    def _add_generic_operate_elements(self, data_attribute, data_attribute_type: MmsDataAttributeType, time_activated: bool, has_check: bool, options: dict):
        control_value = MMSDataAttribute(name=MmsDataAttributeName.CONTROL_VALUE,
                                         data_attribute_type=data_attribute_type,
                                         functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO)
        data_attribute.add_child(control_value)
        self._add_common_operate_elements(data_attribute=data_attribute, time_activated=time_activated, has_check=has_check, options=options)

    def _add_common_operate_elements(self, data_attribute, time_activated: bool, has_check: bool, options: dict):
        if time_activated:
            timed_attribute = MMSDataAttribute(name=MmsDataAttributeName.TIMED_OPERATE,
                                               data_attribute_type=MmsDataAttributeType.TIMESTAMP,
                                               functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO)
            data_attribute.add_child(timed_attribute)

        self._add_originator(data_attribute=data_attribute,
                             name=MmsDataAttributeName.ORIGIN,
                             functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO)

        data_attribute.add_child(MMSDataAttribute(name=MmsDataAttributeName.CONTROL_NUMBER,
                                                  data_attribute_type=MmsDataAttributeType.INT8U,
                                                  functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO))
        data_attribute.add_child(MMSDataAttribute(name=MmsDataAttributeName.CONTROL_TIMESTAMP,
                                                  data_attribute_type=MmsDataAttributeType.TIMESTAMP,
                                                  functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO))
        data_attribute.add_child(MMSDataAttribute(name=MmsDataAttributeName.TEST,
                                                  data_attribute_type=MmsDataAttributeType.BOOLEAN,
                                                  functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO))
        if has_check:
            data_attribute.add_child(MMSDataAttribute(name=MmsDataAttributeName.CHECK,
                                                      data_attribute_type=MmsDataAttributeType.CHECK,
                                                      functional_constraint=MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO))

    def _add_originator(self, data_attribute, name: str, functional_constraint: MMSFunctionalConstraints):
        origin = MMSDataAttribute(name=name, data_attribute_type=MmsDataAttributeType.CONSTRUCTED, functional_constraint=functional_constraint)
        data_attribute.add_child(origin)
        origin.add_child(MMSDataAttribute(name=MmsDataAttributeName.ORIGIN_CATEGORY,
                                          data_attribute_type=MmsDataAttributeType.ENUMERATED,
                                          functional_constraint=functional_constraint))
        origin.add_child(MMSDataAttribute(name=MmsDataAttributeName.ORIGIN_IDENTIFIER,
                                          data_attribute_type=MmsDataAttributeType.OCTET_STRING_64,
                                          functional_constraint=functional_constraint))

    def _add_originator_and_control_number(self, parent, options: dict):
        if options.get("add_origin"):
            self._add_originator(parent, name=MmsDataAttributeName.ORIGIN, functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST)
        # Control Number
        if options.get("add_control_number"):
            parent.add_child(MMSDataAttribute(name=MmsDataAttributeName.CONTROL_NUMBER,
                                              data_attribute_type=MmsDataAttributeType.INT8U,
                                              functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST))

    def _add_control_status_for_analog_control(self, parent, options: dict):
        if options.get("add_origin"):
            self._add_originator(parent, name=MmsDataAttributeName.ORIGIN, functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX)
        # Control Number
        if options.get("add_control_number"):
            parent.add_child(MMSDataAttribute(name=MmsDataAttributeName.CONTROL_NUMBER,
                                              data_attribute_type=MmsDataAttributeType.INT8U,
                                              functional_constraint=MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX))
