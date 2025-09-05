import enum


class MMSFunctionalConstraints(str, enum.Enum):
    NONE = "NONE"

    PROCESS_VALUE_STATUS_ST = "ST"
    PROCESS_VALUE_MEASURAND_MX = "MX"
    PROCESS_COMMAND_BINARY_CO = "CO"
    PROCESS_COMMAND_ANALOG_SP = "SP"
    OPERATE_RECEIVED_OR = "OR"
    CONFIGURATION_CF = "CF"
    DESCRIPTION_DC = "DC"
    SETTINGS_GROUP_ACTIVE_SG = "SG"
    SETTINGS_GROUP_EDITABLE_SE = "SE"

    UNBUFFERED_REPORT = "RP"
    BUFFERED_REPORT = "BR"
    LOG_CONTROL_BLOCK = "LG"
    GOOSE_CONTROL_BLOCK = "GO"

    def get_direction(self) -> str:
        if self in [
            MMSFunctionalConstraints.PROCESS_VALUE_STATUS_ST,
            MMSFunctionalConstraints.PROCESS_VALUE_MEASURAND_MX,
        ]:
            return "monitoring"
        if self in [
            MMSFunctionalConstraints.PROCESS_COMMAND_BINARY_CO,
            MMSFunctionalConstraints.PROCESS_COMMAND_ANALOG_SP
        ]:
            return "control"
        if self in [
            MMSFunctionalConstraints.CONFIGURATION_CF
        ]:
            # TODO: Fix this - should be control? Or "writable"?
            return "monitoring"

        raise ValueError(f"Unknown direction for MMSFunctionalConstraints {self.name}")
