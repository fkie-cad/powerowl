import abc

from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName


class ProtocolInfo(abc.ABC):
    _gid: int = 0

    def __init__(self, protocol_name: ProtocolName):
        self.name = protocol_name
        self._gid = ProtocolInfo._gid
        ProtocolInfo._gid += 1

    def get_protocol_dict(self, as_primitive: bool = False) -> dict:
        return {
            "protocol": self.name.value,
            "protocol_data": self.get_protocol_data_dict(as_primitive)
        }

    def to_dict(self, as_primitive: bool = False) -> dict:
        return {
            "id": self._gid,
            "name": self.name.name if as_primitive else self.name
        }

    def from_dict(self, d: dict):
        self._gid = d["id"]
        self.name = d["name"]

    @abc.abstractmethod
    def get_protocol_data_dict(self, as_primitive: bool = False) -> dict:
        ...

    def generate_data_point_id(self) -> str:
        return f"dp.{self._gid}"
