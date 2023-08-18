import enum


class LinkType(str, enum.Enum):
    DIGITAL = "digital"
    OPTICAL = "optical"
    WIRELESS = "wireless"
