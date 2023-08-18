import enum


class OUs(str, enum.Enum):
    OFFICE = "office"
    CONTROL_CENTER = "control-center"
    OPERATION = "operation"
