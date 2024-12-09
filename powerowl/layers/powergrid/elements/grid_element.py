import abc
import copy
import enum
import warnings
from typing import List, Type, TYPE_CHECKING, Tuple, Optional, Union, Dict

from powerowl.exceptions.grid_value_error import GridValueError
from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo

from powerowl.layers.powergrid.elements.attribute_specification import AttributeSpecification
from powerowl.layers.powergrid.values.grid_value import GridValue
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext
from powerowl.layers.network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo
from powerowl.layers.powergrid.values.grid_value_type import Step

if TYPE_CHECKING:
    from powerowl.layers.powergrid.power_grid_model import PowerGridModel
    from powerowl.layers.powergrid.elements import Bus


class GridElement(abc.ABC):
    prefix: str = "grid-element"
    indices = {}

    @classmethod
    def next_index(cls):
        cls_str = str(cls.__name__)
        index = GridElement.indices.get(cls_str, -1)
        index += 1
        GridElement.indices[cls_str] = index
        return index

    @classmethod
    def update_index(cls, index: int):
        cls_str = str(cls.__name__)
        current_index = GridElement.indices.get(cls_str, -1)
        GridElement.indices[cls_str] = max(index, current_index)

    def __init__(self, create_specification: bool = True, sanitize_specification: bool = False, **kwargs):
        if "index" in kwargs:
            self.index = kwargs.pop("index")
            self.update_index(self.index)
        else:
            self.index = self.next_index()
        self._attributes = {
            GridValueContext.PROPERTY: {},
            GridValueContext.CONFIGURATION: {},
            GridValueContext.MEASUREMENT: {},
            GridValueContext.ESTIMATION: {},
            GridValueContext.GENERIC: {},
        }
        self._data = {}
        if not create_specification:
            return
        specifications = self.get_specifications()
        while len(specifications) > 0:
            spec = specifications.pop(0)
            value = GridValue.from_specification(spec, self)
            self.set_attribute_operator_controllable(spec.context, spec.name, spec.operator_controllable)
            if spec.name in kwargs:
                val = kwargs[spec.name]
                if isinstance(val, GridValue):
                    val = val.raw_get_value()
                value.raw_set_value(val)
            if spec.required and value.value is None and sanitize_specification:
                raise GridValueError(f"Required value {spec.name} missing for {self.get_identifier()}")
            self._attributes[spec.context][spec.name] = value
            if spec.context == GridValueContext.MEASUREMENT:
                spec_est = spec.copy()
                spec_est.context = GridValueContext.ESTIMATION
                spec_est.targets = None
                specifications.append(spec_est)

    def clone(self, clone_index: bool = False):
        kwargs = {}
        if clone_index:
            kwargs["index"] = self.index
        cloned_element = self.__class__(create_specification=False, **kwargs)
        cloned_element._data = copy.deepcopy(self._data)
        cloned_element._attributes = copy.deepcopy(self._attributes)
        print(f"Cloned {self.get_identifier()} as {cloned_element.get_identifier()}")
        return cloned_element

    @staticmethod
    @abc.abstractmethod
    def get_specifications() -> List[AttributeSpecification]:
        ...

    def get_grid_values(self, context: Optional[Union[GridValueContext, List[GridValueContext]]] = None) -> List[Tuple[str, GridValue]]:
        results = []
        if context is not None:
            if isinstance(context, list):
                contexts = context
            else:
                contexts = [context]
        else:
            contexts = [GridValueContext.GENERIC, GridValueContext.PROPERTY, GridValueContext.CONFIGURATION,
                        GridValueContext.MEASUREMENT, GridValueContext.ESTIMATION]
        for context in contexts:
            for attribute_name, attribute in self._attributes[context].items():
                results.append((attribute_name, attribute))
        return results

    def get(self, key, context: GridValueContext) -> GridValue:
        if key in self._attributes[context]:
            return self._attributes[context][key]
        raise KeyError(f"This element ({self.get_identifier()}) has no {context.value} '{key}'")

    def set(self, key, context: GridValueContext, value: GridValue) -> GridValue:
        self._attributes[context][key] = value
        return value

    def get_data(self, key=None, default=None):
        if key is None:
            return self._data
        return self._data.get(key, default)

    def set_data(self, key, value):
        self._data[key] = value

    def rm(self, key, context: GridValueContext) -> bool:
        if self.has(key, context):
            del self._attributes[context][key]
            return True
        return False

    def has(self, key, context: GridValueContext) -> bool:
        try:
            self.get(key, context)
            return True
        except KeyError:
            return False

    def get_generic(self, key) -> GridValue:
        return self.get(key, GridValueContext.GENERIC)

    def get_property(self, key) -> GridValue:
        return self.get(key, GridValueContext.PROPERTY)

    def get_config(self, key) -> GridValue:
        return self.get(key, GridValueContext.CONFIGURATION)

    def get_measurement(self, key) -> GridValue:
        return self.get(key, GridValueContext.MEASUREMENT)

    def get_estimation(self, key) -> GridValue:
        return self.get(key, GridValueContext.ESTIMATION)

    def get_generic_value(self, key, default: Optional = Exception):
        try:
            return self.get_generic(key).value
        except Exception as e:
            if default is not Exception:
                return default
            raise e

    def get_property_value(self, key, default: Optional = Exception):
        try:
            return self.get_property(key).value
        except Exception as e:
            if default is not Exception:
                return default
            raise e

    def get_config_value(self, key, default: Optional = Exception):
        try:
            return self.get_config(key).value
        except Exception as e:
            if default is not Exception:
                return default
            raise e

    def get_measurement_value(self, key, default: Optional = Exception):
        try:
            return self.get_measurement(key).value
        except Exception as e:
            if default is not Exception:
                return default
            raise e

    def get_estimation_value(self, key, default: Optional = Exception):
        try:
            return self.get_estimation(key).value
        except Exception as e:
            if default is not Exception:
                return default
            raise e

    def get_name(self):
        return self.get_generic_value("name")

    def get_readable_name(self):
        return f"{self.__class__.__name__} {self.index}"

    def to_dict(self) -> dict:
        return {
            "type": self.prefix,
            "index": self.index,
            "attributes": self._attributes,
            "data": self._data
        }

    def to_primitive_dict(self, options: Dict) -> dict:
        d = self.to_dict()
        attributes = d["attributes"]
        primitive_attributes = {}
        for attribute_type, attributes_of_type in attributes.items():
            primitive_attributes[attribute_type.name] = {}
            for attribute_name, attribute_value in attributes_of_type.items():
                attribute_value: GridValue
                value = attribute_value.value
                if isinstance(value, enum.Enum):
                    value = value.name
                if isinstance(value, GridElement):
                    value = value.get_identifier()
                if options.get("primitive_attributes", True):
                    primitive_attributes[attribute_type.name][attribute_name] = value
                else:
                    primitive_attributes[attribute_type.name][attribute_name] = {
                        "value": value,
                        "scale": attribute_value.scale.name,
                        "unit": attribute_value.unit.name
                    }
        d["attributes"] = primitive_attributes
        return d

    def from_primitive_dict(self, primitive_dict: dict, power_grid: 'PowerGridModel'):
        """
        Loads this grid element instance based on the given primitive dict and the set of other grid elements
        """
        self._data = primitive_dict.get("data", {})
        for spec in self.get_specifications():
            self._attributes[spec.context][spec.name] = GridValue.from_specification(spec, self)
        # Restore values
        for attribute_type, attributes in primitive_dict["attributes"].items():
            context = GridValueContext[attribute_type]
            for attribute_name, attribute_value in attributes.items():
                try:
                    value = self.get(attribute_name, context)
                except KeyError:
                    warnings.warn(f"{context.value}.{attribute_name} does not exists for {self.get_identifier()} - might be an incompatible PowerOwl version")
                    continue
                try:
                    if issubclass(value.value_type, GridElement):
                        # Reference to another grid element
                        if attribute_value is None:
                            warnings.warn(f"{context.value}.{attribute_name} is None, but should be {value.value_type.__name__}")
                            value.set_value(None)
                        else:
                            element_type, element_id = attribute_value.split(".")
                            element = power_grid.get_element(element_type, int(element_id))
                            value.set_value(element)
                    elif issubclass(value.value_type, enum.Enum):
                        value.set_value(value.value_type[attribute_value])
                    else:
                        value.set_value(attribute_value)
                except TypeError:
                    # issubclass cannot handle typing classes
                    value.set_value(attribute_value)

    def from_dict(self, d: dict):
        self.index = d["index"]
        self._attributes = d["attributes"]
        self._data = d["data"]

    def __eq__(self, other):
        if isinstance(other, GridElement):
            return other.get_identifier() == self.get_identifier()
        return False

    def __hash__(self):
        return id(self)

    def get_index(self) -> str:
        return self.index

    def get_identifier(self):
        return f"{self.prefix}.{self.index}"

    def get_elements_in_attributes(self) -> List['GridElement']:
        """
        Returns a list of GridElements that are referenced by this element's attributes.
        """
        elements = []
        grid_value: GridValue
        for _, grid_value in self.get_grid_values():
            if isinstance(grid_value.value, GridElement):
                elements.append(grid_value.value)
        return elements

    def is_detachable(self) -> bool:
        try:
            controllable = self.get_property("detachable").value
            return controllable
        except KeyError:
            return False

    def is_controllable(self) -> bool:
        try:
            controllable = self.get_property("controllable").value
            return controllable
        except KeyError:
            return False

    def is_observable(self) -> bool:
        try:
            observable = self.get("observable", GridValueContext.GENERIC).value
            return observable
        except KeyError:
            return False

    def is_attribute_operator_controllable(self, attribute_context: GridValueContext, attribute_name: str):
        return self._data.get("grid_operator_controllability", {}).get(attribute_context, {}).get(attribute_name, True)

    def set_attribute_operator_controllable(self, attribute_context: GridValueContext, attribute_name: str, controllable: bool = True):
        self._data.setdefault("grid_operator_controllability", {}).setdefault(attribute_context, {})[attribute_name] = controllable

    def is_attribute_operator_readable(self, attribute_context: GridValueContext, attribute_name: str):
        return self._data.get("grid_operator_readability", {}).get(attribute_context, {}).get(attribute_name, True)

    def set_attribute_operator_readable(self, attribute_context: GridValueContext, attribute_name: str, readable: bool = True):
        self._data.setdefault("grid_operator_readability", {}).setdefault(attribute_context, {})[attribute_name] = readable

    def get_providers(self, filtered: bool = False) -> List[ProviderInfo]:
        providers = []
        try:
            bus = self.get_property_value("bus")
        except KeyError:
            bus = None

        specifications = self.get_specifications()
        for spec in specifications:
            # Include all present attributes of supported types
            if spec.context not in [GridValueContext.MEASUREMENT, GridValueContext.CONFIGURATION]:
                # Skip static parameters
                continue
            if spec.value_type not in [bool, int, float, Step]:
                # Skip unsupported types
                continue
            if not spec.required and spec.default_value is None:
                # Skip empty & non-required attributes
                continue
            if spec.context == GridValueContext.CONFIGURATION and not self.is_attribute_operator_controllable(spec.context, spec.name):
                # Skip attributes that are actively marked as not operator controllable
                continue
            if spec.context == GridValueContext.MEASUREMENT and not self.is_attribute_operator_readable(spec.context, spec.name):
                # Skip attributes that are actively marked as not operator readable
                continue
            provider = PowerGridProviderInfo()
            if spec.context == GridValueContext.MEASUREMENT:
                if bus is not None and spec.name == "is_connected" and (self.is_observable() or bus.is_observable()):
                    # For the "is_connected" measurement, either the asset or the associated bus has to be observable
                    pass
                elif filtered and not self.is_observable():
                    # Skip monitoring attributes when element is not observable and filtering is enabled
                    continue
                provider.domain = "source"
            elif spec.context == GridValueContext.CONFIGURATION:
                if bus is not None and filtered and spec.name == "connected" and (self.is_controllable() or bus.is_controllable()):
                    # For the "connected" control option, either the asset or the associated bus has to be controllable
                    pass
                elif filtered and not self.is_controllable():
                    # Skip control attributes when element is not controllable and filtering is enabled
                    continue
                provider.domain = "target"
            # Build Power Grid Provider current attribute
            provider.element_id = self.get_identifier()
            provider.attribute_context = spec.context
            provider.attribute_name = spec.name
            provider.attribute_type = spec.value_type
            providers.append(provider)

        return providers

    def get_buses(self) -> List['Bus']:
        return []

    @staticmethod
    def element_class_by_type(element_type: str) -> Type['GridElement']:
        from powerowl.layers.powergrid.elements import (Bus, DcLine, ExtendedWard, ExternalGrid, Generator,
                                                        Impedance, Line, Load, Motor, Shunt, StaticGenerator, Storage,
                                                        Switch, Transformer, Ward)
        candidates = [Bus, DcLine, ExtendedWard, ExternalGrid, Generator, Impedance, Line, Load, Motor, Shunt,
                      StaticGenerator, Storage, Switch, Transformer, Ward]
        for grid_element_class in candidates:
            if grid_element_class.prefix == element_type:
                return grid_element_class
        raise KeyError(f"No grid element class for type {element_type} found")
