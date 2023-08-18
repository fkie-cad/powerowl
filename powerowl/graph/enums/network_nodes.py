import enum


class NetworkNodes(str, enum.Enum):
    NODE = "node"
    SWITCH = "switch"
    HOST = "host"
    ROUTER = "router"
    RTU = "rtu"
    MTU = "mtu"
    LINK = "link"
