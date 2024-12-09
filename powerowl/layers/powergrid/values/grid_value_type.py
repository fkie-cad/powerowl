import abc
from typing import Type, Any

GridValueType = Type


class GridValueWrappedType(abc.ABC):
    value_type = Any


class Step(GridValueWrappedType):
    value_type = int
