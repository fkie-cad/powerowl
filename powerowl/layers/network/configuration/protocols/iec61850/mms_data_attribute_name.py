import enum


class MmsDataAttributeName(str, enum.Enum):
    STATUS_VALUE = "stVal"
    QUALITY = "q"
    TIMESTAMP = "t"
    CONTROL_VALUE = "ctlVal"
    CONTROL_MODEL = "ctlModel"
    CONTROL_TIMESTAMP = "T"

    OPERATE = "Oper"
    CANCEL = "Cancel"
    SELECT_BEFORE_OPERATE = "SBO"
    TIMED_OPERATE = "operTm"
    CHECK = "Check"
    TEST = "Test"
    CONTROL_NUMBER = "ctlNum"
    ORIGIN = "origin"
    ORIGIN_CATEGORY = "orCat"
    ORIGIN_IDENTIFIER = "orIdent"

    MEASUREMENT_VALUE = "mxVal"

    MAGNITUDE = "mag"
    INSTANTANEOUS_MAGNITUDE = "instMag"
    RANGE = "range"

    PERSISTENT = "persistent"

    VALUE_WITH_TRANSIENT_INDICATOR = "valWTr"
    POSITION_VALUE = "posVal"
    TRANSIENT_INDICATOR = "transInd"
    MINIMUM_VALUE = "minVal"
    MAXIMUM_VALUE = "maxVal"

    DESCRIPTION = "d"
    DESCRIPTION_UNICODE = "dU"
    CDC_NAME_SPACE = "cdcNs"
    CDC_NAME = "cdcName"
    DATA_NAME_SPACE = "dataNs"

    # FROZEN_VALUE = "frVal"

    SUBSTITUTION_ENABLED = "subEna"
    SUBSTITUTION_VALUE = "subVal"
    SUBSTITUTION_QUALITY = "subQ"
    SUBSTITUTION_ID = "subID"
    SUBSTITUTION_BLOCK_ENABLED = "blkEna"

