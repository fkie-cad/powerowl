import dataclasses
from typing import TYPE_CHECKING, Optional

from powerowl.graph.model_node import ModelNode

if TYPE_CHECKING:
    from powerowl.power_owl import PowerOwl


@dataclasses.dataclass(eq=False, kw_only=True)
class Facility(ModelNode):
    readable_name: Optional[str] = None

    def derive_ict(self, owl: 'PowerOwl'):
        pass

    def __post_init__(self):
        super().__post_init__()
        if self.readable_name is None:
            self.readable_name = self.name

    def __str__(self):
        return f'"Facility {self.name}"'
