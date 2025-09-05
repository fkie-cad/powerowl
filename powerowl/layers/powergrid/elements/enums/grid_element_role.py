import enum


class GridElementRole(str, enum.Enum):
    """
    Describes the intended role of a grid element that can be used during the derivation process.
    """
    AUTO = "auto"
    HOUSEHOLD = "household"
    JUNCTION_BOX = "junction_box"
    SUBSTATION = "substation"
    MUFF = "muff"
