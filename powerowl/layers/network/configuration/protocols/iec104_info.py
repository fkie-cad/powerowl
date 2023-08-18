from typing import Type

from powerowl.layers.network.configuration.data_point_value import DataPointValue
from powerowl.layers.network.configuration.protocols.protocol_info import ProtocolInfo
from powerowl.layers.network.configuration.protocols.protocol_name import ProtocolName


class IEC104Info(ProtocolInfo):
    def __init__(self):
        super().__init__(ProtocolName.IEC104)
        self.coa: int = -1
        self.ioa: int = -1
        self.cot: int = -1
        self.type_id: int = -1
        self._direction: str = "monitoring"

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction: str):
        direction = direction.lower()
        if direction not in ["monitoring", "control"]:
            raise ValueError("Invalid direction. Must be either 'control' or 'monitoring'")
        self._direction = direction

    def to_dict(self, as_primitive: bool = False) -> dict:
        d = super().to_dict(as_primitive)
        d.update(self.get_protocol_data_dict(as_primitive))
        return d

    def from_dict(self, d: dict):
        super().from_dict(d)
        self.coa = d["coa"]
        self.ioa = d["ioa"]
        self.cot = d["cot"]
        self.type_id = d["type_id"]
        self.direction = d["direction"]

    def get_protocol_data_dict(self, as_primitive: bool = False) -> dict:
        return {
            "coa": self.coa,
            "ioa": self.ioa,
            "cot": self.cot,
            "type_id": self.type_id,
            "direction": self.direction
        }

    def generate_data_point_id(self):
        # TODO: Prefix with protocol name?
        return f"{self.coa}.{self.ioa}"

    def set_type_id_by_value_type(self, data_type: Type[DataPointValue]) -> bool:
        if data_type == float:
            if self.direction == "monitoring":
                self.type_id = 13
                return True
            else:
                self.type_id = 50
                return True

        if data_type == int:
            if self.direction == "monitoring":
                self.type_id = 9
                return True
            else:
                self.type_id = 48
                return True

        if data_type == bool:
            if self.direction == "monitoring":
                self.type_id = 1
                return True
            else:
                self.type_id = 45
                return True

        return False

    def derive_cot(self) -> bool:
        if self.direction == "control":
            self.cot = 6
            return True

        if self.type_id in [13, 51]:
            self.cot = 1
            return True

        self.cot = 3
        return True
