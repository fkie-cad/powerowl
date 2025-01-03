import enum


class MMSTriggerOptions(str, enum.Enum):
    DATA_CHANGED = "TRG_OPT_DATA_CHANGED"
    QUALITY_CHANGED = "TRG_OPT_QUALITY_CHANGED"
    DATA_UPDATE = "TRG_OPT_DATA_UPDATE"
    INTEGRITY = "TRG_OPT_INTEGRITY"
    GI = "TRG_OPT_GI"
    TRANSIENT = "TRG_OPT_TRANSIENT"
