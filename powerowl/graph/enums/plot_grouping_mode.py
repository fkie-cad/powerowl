import enum


class PlotGroupingMode(enum.Enum):
    PER_LAYER = "LAYER"
    PER_FACILITY = "FACILITY"
    PER_FACILITY_PER_LAYER = "BOTH"
    PER_FACILITY_OR_LAYER = "FACILITY_OR_LAYER"
    BY_TYPE = "TYPE"
