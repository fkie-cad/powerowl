import enum


class EdgeType(str, enum.Enum):
    DEFAULT = "default"
    PHYSICAL_POWER = "physical-power"
    NETWORK_LINK = "link"
    MEMBERSHIP = "membership"
    LOGICAL = "logical"
    RELATION = "relation"
    ASSOCIATED = "associated"
    CONTROL = "control"
    RESPONSIBLE = "responsible"
    IEC104_CONNECTION = "iec104"
    IP_SUBNET = "ip-subnet"

    def get_short_name(self) -> str:
        return self.value[:3]

    def get_color(self) -> str:
        if self in [EdgeType.PHYSICAL_POWER]:
            return "#101010"
        if self in [EdgeType.NETWORK_LINK]:
            return "#32416e"
        if self in [EdgeType.CONTROL, EdgeType.RESPONSIBLE]:
            return "#ffc296"
        if self in [EdgeType.LOGICAL, EdgeType.RELATION]:
            return "#a9b9cc"
        if self in [EdgeType.ASSOCIATED]:
            return "#eeeeee"
        if self in [EdgeType.MEMBERSHIP]:
            return "#dddddd"
        if self in [EdgeType.IEC104_CONNECTION]:
            return "#306e41"
        return "#aaaaaa"
