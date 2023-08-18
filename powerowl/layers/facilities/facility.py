import dataclasses
from typing import TYPE_CHECKING

from powerowl.graph.model_node import ModelNode

if TYPE_CHECKING:
    from powerowl.power_owl import PowerOwl


@dataclasses.dataclass(eq=False, kw_only=True)
class Facility(ModelNode):
    def derive_ict(self, owl: 'PowerOwl'):
        pass

    def __str__(self):
        return f'"Facility {self.name}"'
