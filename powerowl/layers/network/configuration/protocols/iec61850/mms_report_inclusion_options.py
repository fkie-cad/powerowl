import enum


class MMSReportInclusionOptions(str, enum.Enum):
    SEQ_NUM = "RPT_OPT_SEQ_NUM"
    TIME_STAMP = "RPT_OPT_TIME_STAMP"
    REASON_FOR_INCLUSION = "RPT_OPT_REASON_FOR_INCLUSION"

