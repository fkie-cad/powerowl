import enum


class ProcessorReturnAction(enum.Enum):
    NONE = "none"
    ITERATION_REQUIRED = "iteration_required"
    KEEP_SIMULATING = "keep_simulating"
