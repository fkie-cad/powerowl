from typing import Any, List, Union

import igraph
import networkx as nx

from powerowl.graph.enums import Layers, EdgeType
from powerowl.layers.powergrid.elements import Bus
from .elements.grid_annotator import GridAnnotator
from .elements.grid_asset import GridAsset
from .elements.grid_edge import GridEdge
from .elements.grid_element import GridElement
from .power_grid_model_builder import PowerGridModelBuilder
from .values.grid_value import GridValue
from .values.grid_value_context import GridValueContext


class PowerGridModel:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.elements = {}
        self.builder = PowerGridModelBuilder(self)

    def simulate(self) -> bool:
        raise NotImplementedError()

    def estimate(self, **kwargs) -> bool:
        """
        Estimates the likely actual current grid state based on the set estimation measurements.
        """
        raise NotImplementedError()

    def to_dict(self) -> dict:
        return {
            "kwargs": self._kwargs,
            "elements": self.elements.copy()
        }

    def clone(self) -> 'PowerGridModel':
        """
        Clones this model (recursively)
        """
        clone = self.__class__()
        clone.from_primitive_dict(self.to_primitive_dict())
        return clone

    def to_primitive_dict(self) -> dict:
        d = self.to_dict()
        elements = d["elements"]
        primitive_elements = {}
        for e_type, elements_of_type in elements.items():
            primitive_elements[e_type] = {}
            for e_id, element in elements_of_type.items():
                primitive_elements[e_type][e_id] = element.to_primitive_dict()
        d["elements"] = primitive_elements
        return d

    def from_primitive_dict(self, power_grid_dict: dict):
        self._kwargs = power_grid_dict.get("kwargs", {})
        self.elements = {}
        primitive_dicts = {}
        for element_type, elements in power_grid_dict["elements"].items():
            element_cls = GridElement.element_class_by_type(element_type)
            for element_id, element_primitive_dict in elements.items():
                element = element_cls(index=element_id)
                self.elements.setdefault(element_type, {})[element_id] = element
                primitive_dicts[element.get_identifier()] = element_primitive_dict
        for element_id, primitive_dict in primitive_dicts.items():
            self.get_element_by_identifier(element_id).from_primitive_dict(primitive_dict, power_grid=self)

    def get_element(self, element_type: str, element_id: int) -> GridElement:
        return self.elements[element_type][element_id]

    def get_element_by_identifier(self, element_identifier: str) -> GridElement:
        element_type, element_id = element_identifier.split(".")
        return self.get_element(element_type, int(element_id))

    def get_elements_by_type(self, element_type: str) -> List[GridElement]:
        return list(self.elements.get(element_type, {}).values())

    def get_grid_value_by_identifier(self, grid_value_identifier: str) -> GridValue:
        element_type, element_id, grid_value_context, grid_value_name = grid_value_identifier.split(".")
        element = self.get_element(element_type, int(element_id))
        return element.get(grid_value_name, GridValueContext[grid_value_context])

    def get_grid_values(self) -> List[GridValue]:
        grid_values = []
        elements: dict
        for _, elements in self.elements.items():
            element: GridElement
            for element in elements.values():
                for _, grid_value in element.get_grid_values():
                    grid_values.append(grid_value)
        return grid_values

    def from_dict(self, dict_representation: dict):
        self.elements = dict_representation["elements"]
        self._kwargs = dict_representation["kwargs"]

    def from_external(self, external_model: Any):
        raise NotImplementedError()

    def to_external(self) -> Any:
        raise NotImplementedError()

    @staticmethod
    def edge_elements():
        return ["line", "trafo", "trafo3w", "dcline", "impedance"]

    @staticmethod
    def physical_node_elements():
        return ["bus"]

    @staticmethod
    def asset_elements():
        return ["load", "sgen", "gen", "impedance", "shunt", "ward", "storage", "external_grid"]

    @staticmethod
    def edge_associated_elements():
        return ["switch"]

    def get_annotators(self, grid_element: Union[GridEdge, Bus]):
        """
        Returns all GridAnnotators that are associated with the given GridElement.
        """
        if isinstance(grid_element, GridEdge):
            annotators: Set = self.get_annotators(grid_element.get_bus_a())
            annotators = annotators.union(self.get_annotators(grid_element.get_bus_b()))
            results = set()
            for annotator in annotators:
                if annotator.get_associated() == grid_element:
                    results.add(annotator)
            return results
        else:
            results = set()
            for annotator_type in self.edge_associated_elements():
                annotator: GridAnnotator
                for annotator in self.get_elements_by_type(annotator_type):
                    bus_a = annotator.get_bus()
                    associated = annotator.get_associated()
                    if bus_a == grid_element or associated == grid_element:
                        results.add(annotator)
            return results

    def get_bus(self, index) -> Bus:
        if isinstance(index, GridValue):
            index = index.value
        return self.elements["bus"][index]

    def to_nx_graph(self, with_layout: bool = True, **kwargs):
        graph = nx.Graph()
        # Add every grid element as node
        for e_type, elements in self.elements.items():
            elem: GridElement
            for index, elem in self.elements.get(e_type, {}).items():
                graph.add_nodes_from([(elem.get_identifier(), {"element": elem, "e_type": e_type})])

        for e_type in self.edge_elements():
            for index, elem in self.elements.get(e_type, {}).items():
                elem: GridEdge
                bus_a = elem.get_bus_a()
                bus_b = elem.get_bus_b()
                e_id = elem.get_identifier()
                edge_props = {"type": EdgeType.PHYSICAL_POWER}
                graph.add_edges_from([
                    (e_id, bus_a.get_identifier(), edge_props),
                    (e_id, bus_b.get_identifier(), edge_props)
                ])

        for e_type in self.asset_elements():
            for index, elem in self.elements.get(e_type, {}).items():
                elem: GridAsset
                bus = elem.get_bus()
                edge_props = {"type": EdgeType.PHYSICAL_POWER}
                graph.add_edges_from([(elem.get_identifier(), bus.get_identifier(), edge_props)])

        for e_type in self.edge_associated_elements():
            for index, elem in self.elements.get(e_type, {}).items():
                elem: GridAnnotator
                # Physical to Bus
                bus = elem.get_bus()
                edge_props = {"type": EdgeType.PHYSICAL_POWER}
                graph.add_edges_from([(elem.get_identifier(), bus.get_identifier(), edge_props)])
                # Physical Association to second element
                edge_props = {"type": EdgeType.PHYSICAL_POWER}
                second_element = elem.get_associated()
                graph.add_edges_from([(elem.get_identifier(), second_element.get_identifier(), edge_props)])
                # Check if physical edge from second_element to bus exists and replace it with a logical one
                if graph.has_edge(bus.get_identifier(), second_element.get_identifier()):
                    nx.set_edge_attributes(
                        graph,
                        {(bus.get_identifier(), second_element.get_identifier()): {"type": EdgeType.LOGICAL}}
                    )

        for node, props in graph.nodes.items():
            if props.get("e_type") == "bus":
                props["color"] = "#880000"
            elif props.get("e_type") in ["line"]:
                props["color"] = "#FF8800"
            elif props.get("e_type") in ["trafo"]:
                props["color"] = "#884400"
            else:
                props["color"] = "#1f78b4"

        if with_layout:
            graph = self.layout_graph(graph, **kwargs)

        return graph

    def layout_graph(self, graph, y_factor: float = 3, y_reverse: bool = True):
        """
        Uses igraph to calculate a Reingold-Tilford layout.
        :param graph: The power grid graph
        :param y_factor: Spacing factor to apply for busses in y-dimension
        :param y_reverse: Whether to flip the layout vertically.
        :return: The power grid graph with adjusted position attributes (pos).
        """
        g = graph.copy()
        if y_reverse:
            y_factor = -y_factor
        # Remove equipment nodes
        node_ids = list(g.nodes.keys())
        root = None
        roots = []
        for n in node_ids:
            node = g.nodes[n]
            if node.get("e_type") in self.asset_elements():
                if root is None and node["e_type"] == "external_grid":
                    root = node["element"].get_bus()
                g.remove_node(n)
        if root is not None:
            root = root.get_identifier()
        # Remove non-physical edges
        edges = list(g.edges.keys())
        for e in edges:
            edge = g.edges[e]
            if edge.get("type") != EdgeType.PHYSICAL_POWER:
                g.remove_edge(e[0], e[1])

        igraph_graph = igraph.Graph.from_networkx(g)
        igraph_graph.vs["name"] = igraph_graph.vs["_nx_name"]

        if root is not None:
            v = igraph_graph.vs.find(name=root)
            root = v.index
            roots = [root]

        layout = igraph_graph.layout("rt", root=roots)
        for i, coords in enumerate(layout.coords):
            n_name = igraph_graph.vs[i]["name"]
            pos = (coords[0], coords[1] * y_factor)
            graph.nodes[n_name]["pos"] = pos
            element = graph.nodes[n_name]["element"]
            if isinstance(element, Bus):
                position = element.get_property("position")
                position.set_value(pos)
        vertical_split = -3
        offset_y = vertical_split / 3
        gap_x = 0.2
        nodes_per_bus = {}
        for n in node_ids:
            node = graph.nodes[n]
            if node.get("e_type") in self.asset_elements():
                node["sub-layer"] = Layers.POWER_GRID_ASSETS
                bus_id = node["element"].get_bus().get_identifier()
                nodes_per_bus.setdefault(bus_id, []).append(n)
            else:
                node["sub-layer"] = Layers.POWER_GRID_CORE
        for bus_id, node_ids in nodes_per_bus.items():
            bus_pos = graph.nodes[bus_id]["pos"]
            asset_count = len(node_ids)
            x = bus_pos[0] - (((asset_count - 1) * gap_x) / 2)
            y = bus_pos[1] + offset_y
            if asset_count % 2 == 1:
                x -= gap_x / 2
            for node_id in node_ids:
                pos = (x, y)
                graph.nodes[node_id]["pos"] = pos
                x += gap_x
        return graph
