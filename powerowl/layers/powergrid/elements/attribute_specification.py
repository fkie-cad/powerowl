import dataclasses
from typing import Any, Optional, List, Tuple

from ..values.grid_value_context import GridValueContext
from ..values.grid_value_type import GridValueType
from ..values.units.scale import Scale
from ..values.units.unit import Unit


@dataclasses.dataclass
class AttributeSpecification:
    name: str
    context: GridValueContext
    value_type: GridValueType
    default_value: Any
    unit: Unit = Unit.NONE
    scale: Scale = Scale.NONE
    required: bool = True
    pp_column: Optional[str] = None
    simulation_context: Optional[GridValueContext] = None
    related: Optional[List[Tuple[GridValueContext, str]]] = None

    def copy(self):
        return dataclasses.replace(self)
