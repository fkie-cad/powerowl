import builtins
import enum
import importlib
from typing import Any, TYPE_CHECKING, Union, Optional, Dict

import networkx as nx

from powerowl.exceptions.graph_decoding_error import GraphDecodingError
from powerowl.exceptions.graph_encoding_error import GraphEncodingError


if TYPE_CHECKING:
    from powerowl.graph import MultiLayerGraph


class GraphDecoder:
    """
    This class allows to import a JSON representation of a MultiLayerGraph created by the GraphEncoder.
    """
    def __init__(self, encoded_graph: dict, mlg: 'MultiLayerGraph'):
        self.encoded_graph = encoded_graph
        self.mlg = mlg
        self._resolved_references = {}
        self._unresolved_references = {}

    def decode(self) -> 'MultiLayerGraph':
        self._resolved_references = {}
        self._unresolved_references = {}
        # Restore references
        self._restore_references()
        # Restore MultiLayerGraph
        self.mlg.graph = nx.Graph()
        ## Create Layers
        layers = self._resolve_data_recursively(self.encoded_graph["layers"], "layers")
        self.mlg._layers = {}
        for layer in layers:
            self.mlg.import_layer(layer)
        ## Add Nodes
        nodes = self._resolve_data_recursively(self.encoded_graph["nodes"], "nodes")
        for model_node in nodes:
            self.mlg.add_node(model_node=model_node)
        ## Add Edges
        edges = self._resolve_data_recursively(self.encoded_graph["edges"], "edges")
        for model_edge in edges:
            self.mlg.add_edge(model_edge=model_edge)
        ## Restore layer objects
        if "layer_objects" in self.encoded_graph:
            layer_objects = self._resolve_data_recursively(self.encoded_graph["layer_objects"], "layer_objects")
            for layer_name, layer_object in layer_objects.items():
                self.mlg.add_layer_object(layer_name, layer_object)
        return self.mlg

    def _restore_references(self):
        self._unresolved_references = self.encoded_graph.get("$references")
        if self._unresolved_references is None or not isinstance(self._unresolved_references, dict):
            raise GraphDecodingError("Cannot decode MultiLayerGraph: No $references found")
        for identifier, reference in self._unresolved_references.items():
            self._resolve_reference_recursively(identifier, reference, identifier)

    def _resolve_reference(self, identifier, context: str = "") -> object:
        return self._resolve_reference_recursively(identifier=identifier, reference=None, context=context)

    def _resolve_reference_recursively(self, identifier, reference: Optional[Dict] = None, context: str = "") -> object:
        if identifier in self._resolved_references:
            return self._resolved_references[identifier]

        if reference is None:
            if identifier not in self._unresolved_references:
                raise GraphDecodingError(f"Unknown reference: {identifier}")
            reference = self._unresolved_references[identifier]
            return self._resolve_reference_recursively(identifier, reference, context)
        fqcn = reference["$class"]
        cls = self._get_class_by_fqcn(fqcn)

        if set(reference.keys()).issubset({"$class", "$sub_types"}):
            # TODO: Handle $sub_types?
            return cls

        if issubclass(cls, enum.Enum):
            enum_name = reference["$name"]
            o = cls[enum_name]
            self._resolved_references[identifier] = o
            return o

        # Create placeholder reference / pointer
        o = cls()
        self._resolved_references[identifier] = o
        restore_method_name = reference["$restore_method"]
        restore_arguments = reference["$restore_args"]
        resolved_arguments = self._resolve_data_recursively(restore_arguments, f"{context}.{o.__class__.__name__}")
        if not callable(getattr(o, restore_method_name)):
            raise GraphDecodingError(f"Cannot decode {fqcn}. Restore method {restore_method_name} not found")
        getattr(o, restore_method_name)(resolved_arguments)
        return o

    def _resolve_data_recursively(self, argument, context: str) -> Any:
        if argument is None:
            return None
        if isinstance(argument, (bool, int, float, str)):
            return argument
        if isinstance(argument, (list, tuple, set)):
            resolved_list = [self._resolve_data_recursively(entry, f"{context}.{i}")
                             for i, entry in enumerate(argument)]
            return argument.__class__(resolved_list)
        if isinstance(argument, dict):
            identifier = argument.get("$ref")
            if identifier is not None:
                # Referenced object - resolve
                if len(argument) == 1:
                    return self._resolve_reference(identifier, context)
                raise GraphDecodingError("$ref found, but reference dictionary has multiple entries")
            if "$class" in argument and "$elements" in argument:
                # Dictionary encoding an iterable
                fqcn = argument["$class"]
                elements = argument["$elements"]
                cls = self._get_class_by_fqcn(fqcn)
                return cls(self._resolve_data_recursively(elements, f"{context}.{cls.__name__}"))
            # Normal Dictionary - resolve recursively
            return {k: self._resolve_data_recursively(v, f"{context}.{k}") for k, v in argument.items()}
        raise GraphDecodingError(f"Failed to decode data: Unsupported argument type {argument.__class__}")

    @staticmethod
    def _get_class_by_fqcn(fqcn: str):
        split = fqcn.split(".")
        module = ".".join(split[:-1])
        cls = split[-1]
        if module == "":
            module = builtins
        else:
            try:
                module = importlib.import_module(module)
            except Exception:
                raise GraphDecodingError(f"Cannot import referenced class module {module}")
        class_object = getattr(module, cls)
        return class_object
