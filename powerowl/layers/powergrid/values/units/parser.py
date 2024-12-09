import enum
from typing import Tuple, Any, Callable, List, Type, Dict

from powerowl.layers.powergrid.values.units.scale import Scale
from powerowl.layers.powergrid.values.units.unit import Unit


class Parser:
    @staticmethod
    def parse(value_with_unit: str) -> Tuple[Any, Scale, Unit]:
        value, unit_with_scale = Parser.extract_unit_with_scale(value_with_unit)
        units_dict = Parser.get_unit_symbols()
        scales_dict = Parser.get_scale_prefixes()
        units_key_order = sorted(units_dict, key=lambda k: len(units_dict[k]), reverse=True)
        scales_key_order = sorted(scales_dict, key=lambda k: len(scales_dict[k]), reverse=True)
        unit = Unit.NONE
        scale = Scale.BASE
        # Extract unit
        for unit_key in units_key_order:
            unit_symbol = units_dict[unit_key]
            if unit_with_scale.endswith(unit_symbol):
                unit = Unit[unit_key]
                unit_with_scale = unit_with_scale.removesuffix(unit_symbol)
                break
        # Extract scale
        for scale_key in scales_key_order:
            scale_prefix = scales_dict[scale_key]
            if unit_with_scale.endswith(scale_prefix):
                scale = Scale[scale_key]
                break
        return value, scale, unit

    @staticmethod
    def get_unit_symbols():
        return Parser._get_map(Unit, "get_symbol")

    @staticmethod
    def get_scale_prefixes():
        return Parser._get_map(Scale, "get_prefix")

    @staticmethod
    def _get_map(enumeration: Type[enum.Enum], method_name: str) -> Dict[str, str]:
        results = {}
        for e in enumeration:
            method = getattr(e, method_name)
            results[e.name] = method()
        return results

    @staticmethod
    def extract_unit_with_scale(value) -> Tuple[float, str]:
        numeric = '0123456789-.'
        i = 0
        for i, c in enumerate(value):
            if c not in numeric:
                break
        number = float(value[:i])
        unit_with_scale = value[i:].lstrip()
        return number, unit_with_scale
