import abc
import enum
import time
import traceback
import warnings
from typing import Optional, Any, Dict, Callable, TYPE_CHECKING, List, Tuple, Union

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
                 related: Optional[List[Tuple[GridValueContext, str]]] = None,
                 source: Optional[Tuple[GridValueContext, str]] = None,
                 targets: Optional[List[Union[Tuple[GridValueContext, str], Tuple[GridValueContext, str, Callable]]]] = None,
                 lower_limit: Optional[float] = None,
                 upper_limit: Optional[float] = None):
        self._grid_element = grid_element

        self._is_locked = False
        self._is_frozen = False
        self._frozen_value = None

        self._lower_limit = lower_limit
        self._upper_limit = upper_limit

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
        self._source = source
        self._targets: List[Tuple[GridValueContext, str]] = []
        if targets is not None:
            self._targets = targets
        self._on_set_callbacks = []
        self._on_before_read_callbacks = []
        self._on_state_change_callbacks = []
        self.simulator_context: Any = simulator_context
        self.value_simulator_context: Optional[GridValueContext] = None
        self.set_value(value, set_targets=False)

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
        return self.get_value()

    @value.setter
    def value(self, value):
        self.set_value(value)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source

    @property
    def targets(self) -> List[Tuple[GridValueContext, str]]:
        return self._targets

    def lock(self):
        """
        Activates lock mode for this GridValue.
        If locked, all subsequent writings to this grid value will be blocked.
        The set_value and set_raw_value methods allow to temporarily override the lock.
        """
        if self.is_locked:
            return
        self._is_locked = True
        self._trigger_on_state_change()

    def unlock(self):
        if not self.is_locked:
            return
        self._is_locked = False
        self._trigger_on_state_change()

    @property
    def is_locked(self):
        return self._is_locked

    def freeze(self, frozen_value):
        """
        Activates freezing mode for this GridValue.
        If frozen, all subsequent readings of this grid value will return the given frozen value.
        The get_value and get_raw_value methods allow to temporarily override the freeze
        """
        self._frozen_value = frozen_value
        self._is_frozen = True
        self._trigger_on_state_change()
        self._trigger_on_set(self.raw_get_value(True))

    def unfreeze(self):
        """
        Deactivates freezing mode.
        """
        if not self.is_frozen:
            return
        self._is_frozen = False
        frozen_value = self._frozen_value
        self._frozen_value = None
        self._trigger_on_state_change()
        self._trigger_on_set(frozen_value)

    @property
    def is_frozen(self):
        return self._is_frozen

    def get_frozen_value(self):
        return self._frozen_value

    def get_value(self, override_freeze: bool = False, target_scale: Scale = Scale.NONE) -> Any:
        for callback in self._on_before_read_callbacks:
            callback(self)

        if self.source is not None:
            if self._grid_element is None:
                warnings.warn(f"GridElement not assigned - cannot read source value of {self.get_identifier()}")
            else:
                try:
                    self._value = self._grid_element.get(self.source[1], self.source[0]).get_value(override_freeze)
                except KeyError:
                    warnings.warn(f"{self._grid_element.get_identifier()}.{self.source[0]}.{self.source[1]} not found")

        value = self.raw_get_value(override_freeze=override_freeze)
        if target_scale != Scale.NONE:
            value = target_scale.from_scale(value, self.scale)
        return value

    def get_grid_element(self) -> Optional['GridElement']:
        return self._grid_element

    def get_identifier(self) -> str:
        return f"{self._grid_element.get_identifier()}.{self._value_context.value}.{self.name}"

    def raw_get_value(self, override_freeze: bool = False):
        """
        Returns the current value of this instance, but does not trigger any callbacks.
        """
        if self.is_frozen and not override_freeze:
            return self._frozen_value
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

    def add_on_state_change_callback(self, callback: Callable[['GridValue'], None], as_first: bool = False):
        """
        Adds a callback that is called after the grid values state has been changed, e.g.,
        when it gets frozen or locked.
        """
        if callback not in self._on_state_change_callbacks:
            if as_first:
                self._on_state_change_callbacks.index(0, callback)
            else:
                self._on_state_change_callbacks.append(callback)

    def _trigger_on_state_change(self):
        for callback in self._on_state_change_callbacks:
            callback(self)

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

    def raw_set_value(self, value, timestamp: Optional[float] = None, value_scale: Optional[Scale] = None, override_lock: bool = False) -> bool:
        """
        Updates the wrapped value, but does not trigger any callback.
        If this GridValue is locked, no changes are applied unless override_lock is True.
        """
        if self.is_locked and not override_lock:
            return False
        self._validate_type(value)
        if timestamp is None:
            timestamp = time.time()
        if value_scale is not None and value_scale != Scale.NONE:
            value = self._scale.from_scale(value, value_scale)
        self._last_updated = timestamp
        changed = False
        try:
            if self.raw_get_value(override_freeze=True) != value:
                changed = True
                self._last_changed = timestamp
        except ValueError:
            changed = True
        self._value = value
        return changed

    def set_value(self,
                  value: Any,
                  timestamp: Optional[float] = None,
                  value_scale: Optional[Scale] = None,
                  override_lock: bool = False,
                  raise_on_out_of_bounds: bool = True,
                  set_targets: bool = True) -> bool:
        """
        Updates the wrapped value
        :param value: The new value to set
        :param timestamp: The timestamp corresponding to the value. Automatically resolves to the current time.
        :param value_scale: An optional scale of the given value.
        :param override_lock: Whether to override a potential active value lock
        :param raise_on_out_of_bounds: Whether to raise an exception if value limits exist and the new value exceeds this limit
        :param set_targets: Whether to set grid values referenced by the targets attribute
        :return: True iff the new value differs from the formerly set one.
        """
        # Check limits
        if value is not None and self._lower_limit is not None:
            if value < self._lower_limit:
                if raise_on_out_of_bounds:
                    raise ValueError(f"Value {value} exceeds lower bound of {self._lower_limit}")
                return False
        if value is not None and self._upper_limit is not None:
            if value > self._upper_limit:
                if raise_on_out_of_bounds:
                    raise ValueError(f"Value {value} exceeds upper bound of {self._upper_limit}")
                return False

        # Check targets
        if set_targets:
            for target in self.targets:
                try:
                    target_grid_value = self.get_grid_element().get(target[1], target[0])
                    target_value = value
                    if len(target) == 3:
                        # If tuple has a third argument, this is the rewrite function
                        target_value = target[2](value)
                    target_grid_value.set_value(target_value)
                except KeyError:
                    warnings.warn(f"Target value {target} not found")

        old_value = self.raw_get_value()
        changed = self.raw_set_value(value, timestamp, value_scale, override_lock=override_lock)
        self._trigger_on_set(old_value)
        return changed

    def _trigger_on_set(self, old_value):
        for callback in self._on_set_callbacks:
            try:
                callback(self, old_value, self.raw_get_value())
            except Exception as e:
                warnings.warn(f"Error during on_set_callback:\n {traceback.format_exc()}")

    def to_dict(self, with_history: bool = False):
        d = {
            "name": self._name,
            "keep_history": self._keep_history,
            "value_type": self._value_type,
            "value_context": self._value_context,
            "value_simulator_context": self.value_simulator_context,
            "scale": self._scale,
            "unit": self._unit,
            "value": self.raw_get_value(override_freeze=True),
            "is_locked": self._is_locked,
            "is_frozen": self._is_frozen,
            "frozen_value": self._frozen_value,
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
        self.raw_set_value(d.get("value"), override_lock=True)
        self._is_locked = d.get("is_locked", False)
        self._is_frozen = d.get("is_frozen", False)
        self._frozen_value = d.get("frozen_value", None)
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
            related=specification.related,
            source=specification.source,
            targets=specification.targets,
            lower_limit=specification.lower_bound,
            upper_limit=specification.upper_bound
        )

    def __hash__(self):
        return id(self)
