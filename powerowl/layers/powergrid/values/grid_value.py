import abc
import enum
import time
from typing import Optional, Any, Dict, Callable, TYPE_CHECKING, List, Tuple

from powerowl.exceptions.grid_value_error import GridValueError
from .grid_value_context import GridValueContext
from .grid_value_type import GridValueType
from .units.scale import Scale
from .units.unit import Unit
from ..elements.attribute_specification import AttributeSpecification

if TYPE_CHECKING:
    from powerowl.layers.powergrid.elements.grid_element import GridElement


class GridValue:
    _enable_type_checking: bool = True

    def __init__(self,
                 grid_element: Optional['GridElement'],
                 name: str = None,
                 value_type: GridValueType = Any,
                 value_context: GridValueContext = GridValueContext.GENERIC,
                 value: Any = None,
                 unit: Unit = Unit.NONE,
                 scale: Scale = Scale.NONE,
                 keep_history: bool = False,
                 simulator_context: Any = None,
                 related: Optional[List[Tuple[GridValueContext, str]]] = None):
        self._grid_element = grid_element
        self._history = {}
        self._keep_history = keep_history
        self._name = name
        self._value = None
        self._value_type = value_type
        self._value_context = value_context
        self._scale = scale
        self._unit = unit
        self._last_changed = time.time()
        self._last_updated = time.time()
        self._related_attributes: List[Tuple[GridValueContext, str]] = []
        if related is not None:
            self._related_attributes = related
        self._on_set_callbacks = []
        self._on_before_read_callbacks = []
        self.simulator_context: Any = simulator_context
        self.value_simulator_context: Optional[GridValueContext] = None
        self.set_value(value)

    @property
    def value_type(self) -> GridValueType:
        return self._value_type

    @property
    def value_context(self) -> GridValueContext:
        return self._value_context

    def __repr__(self):
        unit_scale_str = ""
        if self._scale is not None:
            unit_scale_str += self._scale.get_prefix()
        if self._unit is not None:
            unit_scale_str += self._unit.get_symbol()
        return f"{repr(self._value)} {unit_scale_str}".strip()

    @property
    def unit(self):
        return self._unit

    @property
    def scale(self):
        return self._scale

    @property
    def name(self):
        return self._name

    def get_related_grid_values(self) -> List['GridValue']:
        related = []
        for (context, name) in self._related_attributes:
            related.append(self._grid_element.get(name, context))
        return related

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def value(self):
        for callback in self._on_before_read_callbacks:
            callback(self)
        return self.raw_get_value()

    @value.setter
    def value(self, value):
        self.set_value(value)

    def get_value(self) -> Any:
        return self.value

    def get_grid_element(self) -> Optional['GridElement']:
        return self._grid_element

    def get_identifier(self) -> str:
        return f"{self._grid_element.get_identifier()}.{self._value_context.value}.{self.name}"

    def raw_get_value(self):
        """
        Returns the current value of this instance, but does not trigger any callbacks.
        """
        return self._value

    def add_on_set_callback(self, callback: Callable[['GridValue', Any, Any], None], as_first: bool = False):
        """
        Adds a callback that is called *after* this instance's value has been updated.
        It provides this instance, the former value and the updated value as arguments.
        """
        if callback not in self._on_set_callbacks:
            if as_first:
                self._on_set_callbacks.insert(0, callback)
            else:
                self._on_set_callbacks.append(callback)

    def add_on_before_read_callback(self, callback: Callable[['GridValue'], None], as_first: bool = False):
        """
        Adds a callback that is called *before* the value of this instance is returned.
        The callback might use set_raw_value to update the value before returning it.
        """
        if callback not in self._on_before_read_callbacks:
            if as_first:
                self._on_before_read_callbacks.insert(0, callback)
            else:
                self._on_before_read_callbacks.append(callback)

    @property
    def last_updated(self) -> float:
        """
        The timestamp of when this value has been updated (not necessarily changed)
        :return:
        """
        return self._last_updated

    @property
    def last_changed(self) -> float:
        """
        The timestamp of when this value has been changed
        :return:
        """
        return self._last_changed

    def to_scale(self, scale: Scale):
        return self._scale.to_scale(self._value, scale)

    def raw_set_value(self, value, timestamp: Optional[float] = None, value_scale: Optional[Scale] = None) -> bool:
        """
        Updates the wrapped value, but does not trigger any callback.
        """
        self._validate_type(value)
        if timestamp is None:
            timestamp = time.time()
        if value_scale is not None and value_scale != Scale.NONE:
            value = self._scale.from_scale(value, value_scale)
        self._last_updated = timestamp
        changed = False
        if self._value != value:
            changed = True
            self._last_changed = timestamp
        self._value = value
        return changed

    def set_value(self, value, timestamp: Optional[float] = None, value_scale: Optional[Scale] = None) -> bool:
        """
        Updates the wrapped value
        :param value: The new value to set
        :param timestamp: The timestamp corresponding to the value. Automatically resolves to the current time.
        :param value_scale: An optional scale of the given value.
        :return: True iff the new value differs from the formerly set one.
        """
        old_value = self.raw_get_value()
        changed = self.raw_set_value(value, timestamp, value_scale)
        for callback in self._on_set_callbacks:
            callback(self, old_value, self.raw_get_value())
        return changed

    def to_dict(self, with_history: bool = False):
        d = {
            "name": self._name,
            "keep_history": self._keep_history,
            "value_type": self._value_type,
            "value_context": self._value_context,
            "value_simulator_context": self.value_simulator_context,
            "scale": self._scale,
            "unit": self._unit,
            "value": self._value,
            "last_updated": self._last_updated,
            "last_changed": self._last_changed
        }
        if with_history:
            d["history"] = self._history
        return d

    def from_dict(self, d: Dict):
        self._keep_history = d.get("keep_history", False)
        self._history = d.get("history", {})
        self._name = d["name"]
        self._scale = d.get("scale")
        self._unit = d.get("unit")
        self._value = d.get("value")
        self._value_context = d.get("value_context")
        self._value_type = d.get("value_type")
        self.value_simulator_context = d.get("value_simulator_context")
        self._last_updated = d.get("last_updated", 0)
        self._last_changed = d.get("last_changed", 0)

    def _validate_type(self, value) -> bool:
        if GridValue._enable_type_checking and value is not None:
            if isinstance(self._value_type, type):
                if issubclass(self._value_type, enum.Enum):
                    if not isinstance(value, self._value_type):
                        raise GridValueError(f"Expected {self.value_type}, got {value}")
        return True

    @staticmethod
    def from_specification(specification: AttributeSpecification, grid_element: Optional['GridElement']):
        return GridValue(
            grid_element=grid_element,
            name=specification.name,
            value_type=specification.value_type,
            value_context=specification.context,
            value=specification.default_value,
            unit=specification.unit,
            scale=specification.scale,
            simulator_context=specification.simulation_context,
            related=specification.related
        )
