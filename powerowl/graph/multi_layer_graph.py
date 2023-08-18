import codecs
import enum
import json
import pickle
import warnings
import threading
from pathlib import Path
from typing import Optional, Any, Dict, Set, Union, TYPE_CHECKING, List, Callable, Type, overload, Tuple, Iterator

import networkx
import networkx as nx

from .constants import MODEL_NODE, MODEL_EDGE
from .enums import EdgeType
from .enums.layer_matching_strategy import LayerMatchingStrategy
from .graph_decoder import GraphDecoder
from .graph_encoder import GraphEncoder
from .graph_layer import GraphLayer
from .graph_layout import GraphLayout
from .model_edge import ModelEdge
from .model_node import ModelNode
from ..exceptions.layer_not_found_exception import LayerNotFoundException

if TYPE_CHECKING:
    from powerowl.power_owl import PowerOwl


class MultiLayerGraph:
    def __init__(self, owl: 'PowerOwl'):
        self._layers: Dict[str, GraphLayer] = {}
        self._layer_objects = {}
        self.graph = nx.Graph()
        self._layout = GraphLayout(self)
        self.owl: 'PowerOwl' = owl
        self._data = {}
        self._rlock = threading.RLock()

    def get(self, item, default=None):
        return self._data.get(item, default)

    def set(self, item, value):
        self._data[item] = value

    def add_layer(self, name: str, z_offset: Optional[float] = None,
                  from_graph: Optional[nx.Graph] = None,
                  parent: Optional['GraphLayer'] = None) -> 'GraphLayer':
        with self._rlock:
            if name in self._layers:
                raise ValueError(f"Layer {name} already exists")
            z_offset = z_offset if z_offset is not None else len(self.get_layers(False))
            layer = GraphLayer(mlg=self, name=name, z_offset=z_offset, parent=parent)
            self._layers[name] = layer
            if from_graph is not None:
                self.merge(graph=from_graph, layer=layer)
            return layer

    def import_layer(self, layer: GraphLayer):
        layer.mlg = self
        if layer.name in self._layers:
            raise ValueError(f"Layer {layer.name} already exists")
        self._layers[layer.name] = layer

    def merge(self, graph: nx.Graph, layer: Union[str, GraphLayer]):
        """
        Merges the multi layer graph with a given (one-layered) nx.Graph into the given layer.
        """
        layer = self.get_layer(layer)
        node_mapping = {}
        for node_name, node in graph.nodes.items():
            model_node = ModelNode(name=node_name, layer=layer)
            for key, value in node.items():
                model_node.attributes[key] = value
            node_mapping[node_name] = model_node
            sub_layer_name = model_node.get("sub-layer")
            if sub_layer_name is not None:
                if not self.has_layer(sub_layer_name):
                    sub_layer = layer.add_sub_layer(sub_layer_name)
                else:
                    sub_layer = self.get_layer(sub_layer_name)
                    if not sub_layer.is_sub_layer(layer):
                        raise ValueError(f"The requested sub-layer {sub_layer_name} exists but is not a sub-layer "
                                         f"of {layer.name}")
                model_node.layer = sub_layer
            self.add_node(model_node=model_node)
        for edge in graph.edges.data("type"):
            node_a = node_mapping[edge[0]]
            node_b = node_mapping[edge[1]]
            edge_type = edge[2]
            if not isinstance(edge_type, EdgeType):
                edge_type = EdgeType.DEFAULT
            model_edge = ModelEdge(node_a=node_a, node_b=node_b, edge_type=edge_type)
            self.add_edge(model_edge=model_edge)

    def insert_layer_above(self, existing_layer: Union[GraphLayer, str], name: str,
                           from_graph: Optional[nx.Graph] = None, shift: float = 1) -> 'GraphLayer':
        e_layer = self.get_layer(existing_layer)
        n_z_index = e_layer.z_offset + shift
        return self.insert_layer(name, z_offset=n_z_index, from_graph=from_graph, shift=shift)

    def insert_layer_below(self, existing_layer: Union[GraphLayer, str], name: str,
                           from_graph: Optional[nx.Graph] = None, shift: float = 1) -> 'GraphLayer':
        e_layer = self.get_layer(existing_layer)
        n_z_index = e_layer.z_offset
        return self.insert_layer(name, z_offset=n_z_index, from_graph=from_graph, shift=shift)

    def insert_layer(self, name: str, z_offset: Optional[float] = 0,
                     from_graph: Optional[nx.Graph] = None, shift: float = 1) -> 'GraphLayer':
        layer = self.add_layer(name, z_offset=z_offset, from_graph=from_graph)
        for existing_layer in self.get_layers(include_sub_layers=False):
            if existing_layer >= layer and existing_layer != layer:
                existing_layer.z_offset += shift
        return layer

    def get_layer(self, layer: Union[str, GraphLayer]):
        if isinstance(layer, GraphLayer):
            return layer
        if not self.has_layer(layer):
            raise LayerNotFoundException(f"Layer {layer} does not exist")
        return self._layers[layer]

    def get_layer_by_uid(self, uid: int) -> GraphLayer:
        for layer in self._layers.values():
            if layer.uid == uid:
                return layer
        raise LayerNotFoundException(f"No Layer with UID {uid} exists")

    def has_layer(self, layer: str):
        return layer in self._layers

    def has_layers(self, *layers: str):
        for layer in layers:
            if not self.has_layer(layer):
                return False
        return True

    def get_layers(self, include_sub_layers: bool = True) -> List['GraphLayer']:
        layers = {layer for layer in self._layers.values() if layer.parent is None or include_sub_layers}
        return sorted(layers, key=lambda layer: layer.z)

    def add_layer_object(self, layer: str, layer_object: Any):
        self._layer_objects[layer] = layer_object

    def get_layer_object(self, layer: str) -> Optional[Any]:
        return self._layer_objects.get(layer)

    def get_layer_graph(self, layer_name: str, include_sub_layers: bool = True,
                        edge_type_filter: Optional[Set[EdgeType]] = None) -> nx.Graph:
        layer = self.get_layer(layer_name)
        return layer.get_layer_graph(include_sub_layers=include_sub_layers, edge_type_filter=edge_type_filter)

    @staticmethod
    def get_node_name(name: Union[str, enum.Enum]):
        if isinstance(name, enum.Enum):
            return f"{name.__class__.__name__}.{name.value}"
        return name

    def get_node_uid(self, node: Union[str, enum.Enum, ModelNode]) -> str:
        if isinstance(node, ModelNode):
            return node.uid
        node = self.get_node_name(node)
        if node in self.graph.nodes:
            return node
        for node_uid in self.graph.nodes.keys():
            node_model = self.get_model_node(node_uid)
            if node_model.name == node:
                return node_model.uid
        raise KeyError(f"No node with ID or name {node} exists")

    def add_node(self, model_node: 'ModelNode'):
        with self._rlock:
            name = model_node.uid
            if model_node.layer is None:
                raise AttributeError("ModelNode layer must not be None")
            self.graph.add_nodes_from([(name, {MODEL_NODE: model_node})])
            return model_node

    def add_edge(self, model_edge: 'ModelEdge'):
        with self._rlock:
            node_a = model_edge.node_a.uid
            node_b = model_edge.node_b.uid
            return self.graph.add_edges_from([(node_a, node_b, {MODEL_EDGE: model_edge})])

    def build_edge(self, node_a: Union[enum.Enum, str, 'ModelNode'], node_b: Union[enum.Enum, str, 'ModelNode'],
                   edge_type: EdgeType = EdgeType.DEFAULT):
        edge = ModelEdge(
            node_a=self.get_model_node(node_a),
            node_b=self.get_model_node(node_b),
            edge_type=edge_type
        )
        return self.add_edge(edge)

    def has_edge(self, node_a, node_b):
        try:
            node_a = self.get_node_uid(node_a)
            node_b = self.get_node_uid(node_b)
        except KeyError:
            return False
        return self.graph.has_edge(node_a, node_b)

    def edge_exists(self, edge: ModelEdge):
        return self.has_edge(edge.node_a, edge.node_b)

    def get_edge(self, node_a, node_b) -> Optional[ModelEdge]:
        if not self.has_edge(node_a, node_b):
            return None
        node_a = self.get_node_uid(node_a)
        node_b = self.get_node_uid(node_b)
        edge_data = self.graph.get_edge_data(node_a, node_b, default={})
        return edge_data.get(MODEL_EDGE)

    def has_path(self,
                 node_a,
                 node_b,
                 allowed_edge_types: Optional[Union[EdgeType, Set[EdgeType]]] = None
                 ) -> bool:
        path = self.get_shortest_path(node_a, node_b, allowed_edge_types)
        return path is not None

    def get_shortest_path(self,
                          node_a,
                          node_b,
                          allowed_edge_types: Optional[Union[EdgeType, Set[EdgeType]]] = None
                          ) -> Optional[List[ModelNode]]:
        node_a = self.get_model_node(node_a)
        node_b = self.get_model_node(node_b)
        edges_to_remove = []
        if allowed_edge_types is not None:
            # Allow using a single edge type as argument
            if isinstance(allowed_edge_types, EdgeType):
                allowed_edge_types = {allowed_edge_types}
            # Select all edges have the wrong type
            edges_to_remove = [e for e in self.graph.edges(data=True)
                               if not e[2][MODEL_EDGE].edge_type in allowed_edge_types]
        # Remove edges
        self.graph.remove_edges_from(edges_to_remove)
        try:
            path = nx.shortest_path(self.graph, node_a.uid, node_b.uid)
        except networkx.exception.NetworkXNoPath:
            return None
        finally:
            # Re-add removed edges
            self.graph.add_edges_from(edges_to_remove)
        model_path = []
        for node in path:
            model_path.append(self.get_model_node(node))
        return model_path

    def get_inter_layer_edges(self,
                              layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.TRANSITIVE
                              ) -> List['ModelEdge']:
        """
        Returns a set of ModelEdge that contains all edges that connect different layers.
        Hereby, the equivalence of layers is defined by the layer_matching_strategy.
        """
        edges = set()
        for edge in self.graph.edges.data(MODEL_EDGE):
            if edge[2] is None:
                raise ValueError(f"No ModelEdge for edge {edge}")
            model_edge: 'ModelEdge' = edge[2]
            if model_edge.is_inter_layer_edge(layer_matching_strategy=layer_matching_strategy):
                edges.add(model_edge)
        return sorted(edges)

    def get_intra_layer_edges(self,
                              layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.TRANSITIVE
                              ) -> List['ModelEdge']:
        """
        Returns a set of ModelEdge that contain all edges that connect nodes of the same layer.
        Hereby, the equivalence of layers is defined by the layer_matching_strategy.
        """
        edges = set()
        for edge in self.graph.edges.data(MODEL_EDGE):
            if edge[2] is None:
                raise ValueError(f"No ModelEdge for edge {edge}")
            model_edge: 'ModelEdge' = edge[2]
            if not model_edge.is_inter_layer_edge(layer_matching_strategy=layer_matching_strategy):
                edges.add(model_edge)
        return sorted(edges)

    def get_neighbors(self,
                      node: Union[str, ModelNode],
                      neighbor_instance_filter: Optional[Set[Type[ModelNode]]] = None
                      ) -> List[ModelNode]:
        neighbors = set()
        node = self.get_model_node(node)
        node_id = node.uid

        for neighbor in self.graph.neighbors(node_id):
            neighbor_model = self.get_model_node(neighbor)
            neighbors.add(neighbor_model)

        if neighbor_instance_filter is not None and len(neighbor_instance_filter) > 0:
            neighbors = {neighbor for neighbor in neighbors if
                         isinstance(neighbor, tuple(neighbor_instance_filter))}
        return sorted(neighbors)

    def get_inter_layer_neighbors(
            self,
            node: Union[str, ModelNode],
            neighbor_exclusion_strategy: LayerMatchingStrategy = LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER,
            other_layers: Optional[Set['GraphLayer']] = None,
            other_layers_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.IS_SUB_LAYER,
            neighbor_instance_filter: Optional[Set[Type[ModelNode]]] = None,
            edge_type_filter: Optional[Set[EdgeType]] = None
    ) -> List[ModelNode]:
        """
        Returns all neighbors of the given node that are on another layer.
        The neighbor_exclusion_strategy defines, how neighbors should be excluded based on their level, i.e., when
            layers of the given node and its neighbor should be treated as equal.
        If other_layers is not None, only neighbors of these layers are returned.
        The other_layers_matching_strategy defines, how the neighbor node's layer should be matched against the
            other_layers, i.e., when the neighbor is treated as part of a given other_layer.
        """
        neighbors = set()
        node = self.get_model_node(node)
        node_id = node.uid

        for neighbor in self.graph.neighbors(node_id):
            neighbor_model = self.get_model_node(neighbor)
            # Skip neighbors that are considered on a matching layer
            if neighbor_exclusion_strategy.matches(node.layer, neighbor_model.layer):
                continue
            # Skip neighbors not matching the other_layers filter
            if other_layers is None:
                neighbors.add(neighbor_model)
            else:
                neighbor_layer = neighbor_model.layer
                for layer in other_layers:
                    if other_layers_matching_strategy.matches(neighbor_layer, layer):
                        neighbors.add(neighbor_model)

        # Filter by neighbor instance type and EdgeType
        neighbors = self._filter_neighbors(node, neighbors, neighbor_instance_filter, edge_type_filter)

        return sorted(neighbors)

    def get_intra_layer_neighbors(
            self,
            node: Union[str, ModelNode],
            layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER,
            neighbor_instance_filter: Optional[Set[Type[ModelNode]]] = None,
            edge_type_filter: Optional[Set[EdgeType]] = None
    ) -> List[ModelNode]:
        """
        Returns all neighbors of the given node that belong to the same layer.
        The layer_matching_strategy defines, how the neighbor node's layer should be matched against the
            other_layers, i.e., when the neighbor is treated as part of the given node's layer.
        If use_top_level_layers is True, all neighbors that share the common top-level layer are returned.
        """
        neighbors = set()
        node = self.get_model_node(node)
        node_id = node.uid
        for neighbor in self.graph.neighbors(node_id):
            neighbor_model = self.get_model_node(neighbor)
            if layer_matching_strategy.matches(node.layer, neighbor_model.layer):
                neighbors.add(neighbor_model)
        
        # Filter by neighbor instance type and EdgeType
        neighbors = self._filter_neighbors(node, neighbors, neighbor_instance_filter, edge_type_filter)
        
        return sorted(neighbors)

    def _filter_neighbors(self,
                          node: ModelNode,
                          neighbors: Set[ModelNode],
                          neighbor_instance_filter: Optional[Set[Type[ModelNode]]],
                          edge_type_filter: Optional[Set[EdgeType]]) -> Set[ModelNode]:
        """
        Filters the given set of neighbors for the given node by their instance as well as the type of the connecting edge.
        """
        if neighbor_instance_filter is not None and len(neighbor_instance_filter) > 0:
            # Apply neighbor instance filter
            neighbors = {neighbor for neighbor in neighbors if isinstance(neighbor, tuple(neighbor_instance_filter))}
        if edge_type_filter is not None and len(edge_type_filter) > 0:
            # Apply edge type filter
            neighbors = {neighbor for neighbor in neighbors if self.get_edge(node, neighbor).edge_type in edge_type_filter}
        return neighbors

    def layout(self):
        with self._rlock:
            return self._layout.layout()

    def plotly_figure(self, **kwargs):
        self._layout.update_config(kwargs)
        return self._layout.plotly_figure()

    def draw(self, **kwargs):
        fig = self.plotly_figure(**kwargs)
        fig.show()

    def get_model_node(self, n: Union[str, ModelNode]) -> 'ModelNode':
        if isinstance(n, ModelNode):
            return n
        n = self.get_node_uid(n)
        return self.graph.nodes[n][MODEL_NODE]
    
    def set_model_node(self, n: str, model_node: ModelNode):
        n = self.get_node_uid(n)
        self.graph.nodes[n][MODEL_NODE] = model_node

    def has_node(self, n: Union[str, ModelNode]) -> bool:
        uid = self.get_node_uid(n)
        return self.graph.has_node(uid)

    def get_subgraph_of_layers(self, layers: Set['GraphLayer'], edge_type_filter: Optional[Set[EdgeType]] = None) -> nx.Graph:
        graph = nx.Graph()
        nodes = self.get_nodes_at_layers(layers)
        edges = self.get_edges_at_layers(layers)
        for model_node in nodes:
            name = model_node.uid
            graph.add_node(name, **{MODEL_NODE: model_node})
        for edge in edges:
            if edge_type_filter is not None and not edge.edge_type in edge_type_filter:
                continue
            graph.add_edge(edge.node_a.uid, edge.node_b.uid, **{MODEL_EDGE: edge})
        return graph

    def get_nodes_at_layers(self,
                            layers: Set['GraphLayer'],
                            layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.IS_SUB_LAYER
                            ) -> List['ModelNode']:
        """
        Get all nodes from this MultiLayerGraph that belong to one of the given layers
        :param layers: The layers to filter nodes by
        :param layer_matching_strategy: How to decide which node layers match the given set of layers.
        :return: A subset of the nodes that match the filter as a dict.
        """
        nodes = set()
        name: str
        model_node: 'ModelNode'
        for name in sorted(self.graph.nodes.keys()):
            model_node = self.get_model_node(name)
            for layer in sorted(layers):
                if layer_matching_strategy.matches(model_node.layer, layer):
                    nodes.add(model_node)
                    break
        return sorted(nodes)

    def get_edges_at_layers(self, layers: Set['GraphLayer'],
                            both_ends_included: bool = True,
                            layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.IS_SUB_LAYER
                            ) -> List['ModelEdge']:
        """
        Get all edges from this MultiLayerGraph that belong to at least one of the given layers.
        If both_ends_included is True, both nodes connected by the edge have to match the layer filter.
        Otherwise, one node suffices.
        :param layers: The layers to filter edges by.
        :param both_ends_included: Whether both nodes connected by an edge have to match the filter.
        :param layer_matching_strategy: Defines how it is decided if nodes belong to a layer.
        """
        edges = set()

        for edge in self.graph.edges.data(MODEL_EDGE):
            name_a = edge[0]
            name_b = edge[1]
            model_edge = edge[2]
            if model_edge is None:
                raise ValueError(f"Invalid edge: {edge}. ModelEdge instance is None")
            model_node_a = self.get_model_node(name_a)
            model_node_b = self.get_model_node(name_b)
            matches = 0
            for model_node in [model_node_a, model_node_b]:
                for layer in layers:
                    if layer_matching_strategy.matches(model_node.layer, layer):
                        matches += 1
                        break
            if matches == 0:
                continue
            if matches == 1 and both_ends_included:
                continue
            edges.add(model_edge)
        return sorted(edges)

    def get_edges(self) -> List[ModelEdge]:
        edges = set()
        for edge in self.graph.edges.data(MODEL_EDGE):
            model_edge = edge[2]
            edges.add(model_edge)
        return sorted(edges)

    def get_nodes(self) -> List[ModelNode]:
        nodes = set()
        for node_id in self.graph.nodes.keys():
            nodes.add(self.get_model_node(node_id))
        return sorted(nodes)

    def remove_node(self, node: Union[str, ModelNode]):
        with self._rlock:
            node = self.get_model_node(node)
            self.graph.remove_node(node.uid)

    def remove_edge(self, edge: ModelEdge):
        with self._rlock:
            self.graph.remove_edge(edge.node_a.uid, edge.node_b.uid)

    #
    # Graph Algorithms
    #
    def bfs(self,
            start_node: Union[str, 'ModelNode'],
            max_depth: Optional[int] = None,
            stop_node_types: Optional[Set[Type['ModelNode']]] = None,
            allowed_node_types: Optional[Set[Type['ModelNode']]] = None,
            stop_edge_types: Optional[Set[Type['EdgeType']]] = None,
            allowed_edge_types: Optional[Set[Type['EdgeType']]] = None,
            layers: Optional[Set['GraphLayer']] = None,
            layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER
            ) -> Iterator[Tuple[ModelNode, Optional[ModelNode], int]]:
        """
        Performs a breadth-first-search on the graph starting at the given node.
        Optionally stops at a certain depth and/or at certain node types.
        Returns a recursive dict where the keys are ModelNodes and the values are lists, recursively containing
        such dictionaries again.
        Yields a Tuple with three elements:
        - The found node
        - Its predecessor (if any, otherwise None)
        - The found node's level
        """
        yield from self._stack_search(
            lambda stack: 0,
            start_node=start_node,
            max_depth=max_depth,
            stop_node_types=stop_node_types,
            allowed_node_types=allowed_node_types,
            stop_edge_types=stop_edge_types,
            allowed_edge_types=allowed_edge_types,
            layers=layers,
            layer_matching_strategy=layer_matching_strategy
        )

    def dfs(self,
            start_node: Union[str, 'ModelNode'],
            max_depth: Optional[int] = None,
            stop_node_types: Optional[Set[Type['ModelNode']]] = None,
            allowed_node_types: Optional[Set[Type['ModelNode']]] = None,
            stop_edge_types: Optional[Set[Type['EdgeType']]] = None,
            allowed_edge_types: Optional[Set[Type['EdgeType']]] = None,
            layers: Optional[Set['GraphLayer']] = None,
            layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER
            ) -> Iterator[Tuple[ModelNode, ModelNode, int]]:
        """
        Performs a depth-first-search on the graph starting at the given node.
        Yields a Tuple with three elements:
        - The found node
        - Its predecessor (if any, otherwise None)
        - The found node's level
        """
        yield from self._stack_search(
            lambda stack: len(stack) - 1,
            start_node=start_node,
            max_depth=max_depth,
            stop_node_types=stop_node_types,
            allowed_node_types=allowed_node_types,
            stop_edge_types=stop_edge_types,
            allowed_edge_types=allowed_edge_types,
            layers=layers,
            layer_matching_strategy=layer_matching_strategy
        )

    def _stack_search(self,
                      pop_index_callback: Callable[[List], int],
                      start_node: Union[str, 'ModelNode'],
                      max_depth: Optional[int] = None,
                      stop_node_types: Optional[Set[Type['ModelNode']]] = None,
                      allowed_node_types: Optional[Set[Type['ModelNode']]] = None,
                      stop_edge_types: Optional[Set[Type['EdgeType']]] = None,
                      allowed_edge_types: Optional[Set[Type['EdgeType']]] = None,
                      layers: Optional[Set['GraphLayer']] = None,
                      layer_matching_strategy: LayerMatchingStrategy = LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER
                      ) -> Iterator[Tuple[ModelNode, ModelNode, int]]:
        start_node = self.get_model_node(start_node)
        seen = set()
        # Second tuple parameter indicates the node's depth
        stack: List[Tuple[ModelNode, int]] = [(start_node, 0)]
        predecessors: Dict[ModelNode, ModelNode] = {}
        while len(stack) > 0:
            model_node, node_depth = stack.pop(pop_index_callback(stack))
            if model_node in seen:
                continue
            seen.add(model_node)
            predecessor = predecessors.get(model_node)
            yield tuple([model_node, predecessor, node_depth])
            if max_depth is not None and node_depth >= max_depth - 1:
                # Maximum depth reached
                continue
            # Find all next hops
            successors = self.get_neighbors(model_node)
            if layers is not None:
                # Exclude nodes that do not belong to the targeted layers
                successors = [
                    successor for successor in successors
                    if any([layer_matching_strategy.matches(successor.layer, layer) for layer in layers])
                ]
            if stop_node_types is not None:
                # Exclude all successors that are an instance of a given stop_node_type
                successors = [
                    successor for successor in successors
                    if not isinstance(successor, tuple(stop_node_types))
                ]
            if allowed_node_types is not None:
                # Exclude all successors that are not an instance of a given include_node_type
                successors = [
                    successor for successor in successors
                    if isinstance(successor, tuple(allowed_node_types))
                ]
            if stop_edge_types is not None:
                # Exclude all nodes connected to by not-allowed edge types
                successors = [
                    successor for successor in successors
                    if self.get_edge(model_node, successor).edge_type not in stop_edge_types
                ]
            if allowed_edge_types is not None:
                # Only include nodes connected to by allowed edge types
                successors = [
                    successor for successor in successors
                    if self.get_edge(model_node, successor).edge_type in allowed_edge_types
                ]
            # add predecessor
            for successor in successors:
                predecessors[successor] = model_node
            # Add successors to stack
            stack.extend([(successor, node_depth+1) for successor in successors])

    #
    # EXPORT / IMPORT
    #
    def encode(self, include_layer_objects: bool = True) -> dict:
        encoder = GraphEncoder(self)
        return encoder.encode(include_layer_objects=include_layer_objects)

    def decode(self, graph_dict: dict):
        decoder = GraphDecoder(encoded_graph=graph_dict, mlg=self)
        decoder.decode()

    def to_dict(self, include_layer_objects: bool = False) -> dict:
        graph_dict = {
            "nodes": self.get_nodes(),
            "edges": self.get_edges()
        }
        layers = [layer.to_dict() for layer in self.get_layers(include_sub_layers=True)]
        d = {
            "graph": graph_dict,
            "layers": layers
        }
        if include_layer_objects:
            d["layer_objects"] = {layer_name: codecs.encode(pickle.dumps(layer_object), "base64").decode()
                                  for layer_name, layer_object in self._layer_objects.items()}
        return d

    def load_dict(self, powerowl_dict: dict):
        self._layer_objects = {}
        if "layer_objects" in powerowl_dict:
            for layer_name, encoded_object in powerowl_dict["layer_objects"].items():
                self._layer_objects[layer_name] = pickle.loads(codecs.decode(encoded_object.encode(), "base64"))
        self.graph = networkx.from_dict_of_dicts(powerowl_dict["graph"])
        layers = [GraphLayer.from_dict(layer_dict, self) for layer_dict in powerowl_dict["layers"]]
        name_set = {layer.solo_name for layer in layers}

        # Fix Layer parent references
        layers_by_uid = {layer.uid: layer for layer in layers}
        for layer in layers:
            parent = layers_by_uid.get(layer.parent)
            if layer.parent >= 0 and parent is None:
                warnings.warn(f"Parent with UID {layer.parent} requested, but not found")
            layer.parent = parent

        if len(name_set) != len(layers):
            warnings.warn("Layers do not have unique names - this unavoidably results in loss of data and a "
                          "corrupted graph!")
        self._layers = {layer.solo_name: layer for layer in layers}
