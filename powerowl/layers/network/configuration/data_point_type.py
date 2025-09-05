import enum


class DataPointType(str, enum.Enum):
    DATA_POINT = "datapoint"
    REPORT_CONTROL_BLOCK_61850 = "61850-rcb"
    DATA_SET_61850 = "61850-dataset"
