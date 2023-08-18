import enum
import typing
import numpy as np
from typing import Any, TYPE_CHECKING, Union, Optional

from powerowl.exceptions.graph_encoding_error import GraphEncodingError


if TYPE_CHECKING:
    from powerowl.graph import MultiLayerGraph


class GraphEncoder:
    """
    This class allows to derive a JSON-serializable dict-representation of the given MultiLayerGraph
    Default data types are preserved, while other objects are transformed via their "to_dict" method.
    To preserve pointers and handle circular references, complex objects are stored in a reference
    table and then referenced in the actual graph structure.

    Supported data types:
    None, bool, int, float, ModelNode, ModelEdge, GridElement
    and any object that provides the to_dict and from_dict methods.
    """
    def __init__(self, mlg: 'MultiLayerGraph'):
        self.mlg = mlg
        self._references = {}

    def encode(self, include_layer_objects: bool = False) -> dict:
        self._references = {}
        encoded_nodes = self._iterencode(self.mlg.get_nodes())
        encoded_edges = self._iterencode(self.mlg.get_edges())
        encoded_layers = self._iterencode(self.mlg.get_layers(True))
        encoded_graph = {
            "$references": self._references,
            "nodes": encoded_nodes,
            "edges": encoded_edges,
            "layers": encoded_layers
        }
        if include_layer_objects:
            encoded_graph["layer_objects"] = {
                layer.name: self._iterencode(self.mlg.get_layer_object(layer=layer.name))
                for layer in self.mlg.get_layers(True)
            }
        return encoded_graph

    def _iterencode(self, o: Any, context: Optional[str] = None):
        if context is None:
            context = ""
        if o is None:
            return None
        elif isinstance(o, enum.Enum):
            # Required for enums that also inherit str or int
            return self._iterencode_object(o, context)
        elif isinstance(o, (str, float, int, bool)):
            return o
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return self._iterencode_list_alike(o.tolist(), context)
        elif isinstance(o, dict):
            return self._iterencode_dict(o, context)
        elif isinstance(o, (list, tuple, set)):
            return self._iterencode_list_alike(o, context)
        else:
            return self._iterencode_object(o, context)

    def _iterencode_list_alike(self, collection: Union[list, tuple, set], context: str):
        """
        Recursively encodes a list-alike object (lists, tuples, sets)
        """
        cls = self._fqcn(collection)
        elements = [self._iterencode(e, f"{context}.{str(i)}") for i, e in enumerate(collection)]
        return {
            "$class": cls,
            "$elements": elements
        }

    def _iterencode_dict(self, d: dict, context: str):
        """
        Recursively encodes a dictionary
        """
        return {k: self._iterencode(v, f"{context}.{k}") for k, v in d.items()}

    def _iterencode_object(self, o: Any, context: str):
        """
        Encodes objects as references.
        If a reference does not yet exist, it is created.
        """
        identifier = str(id(o))
        if identifier not in self._references:
            # Create placeholder reference
            self._references[identifier] = None
            cls = self._fqcn(o)
            dict_representation = None
            if isinstance(o, type):
                dict_representation = {
                    "$class": cls
                }
            elif o.__class__.__module__ == "typing":
                dict_representation = {
                    "$class": repr(o).split("[")[0],
                    "$sub_types": [
                        self._iterencode(sub_type, context) for sub_type in typing.get_args(o)
                    ]
                }
            elif isinstance(o, enum.Enum):
                dict_representation = {
                    "$class": cls,
                    "$name": o.name
                }
            else:
                if callable(getattr(o, "to_dict", None)) and callable(getattr(o, "from_dict", None)):
                    dict_representation = {
                        "$class": cls,
                        "$restore_method": "from_dict",
                        "$restore_args": self._iterencode_dict(o.to_dict(), f"{context}.{o.__class__.__name__}")
                    }
            if dict_representation is None:
                raise GraphEncodingError(f"{context}\nCannot encode object of type {cls}.")
            self._references[identifier] = dict_representation
        return self._build_reference(identifier)

    @staticmethod
    def _build_reference(identifier) -> dict:
        return {"$ref": str(identifier)}

    @staticmethod
    def _fqcn(instance_or_class) -> str:
        """
        Returns the fully-qualified class name of a given object.
        """
        if isinstance(instance_or_class, type):
            cls = instance_or_class
        else:
            cls = instance_or_class.__class__
        # cls = instance_or_class.__class__
        module = cls.__module__
        if module == "builtins":
            return cls.__qualname__
        return f"{module}.{cls.__qualname__}"
