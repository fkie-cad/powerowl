import copy
import math
import warnings
from typing import TYPE_CHECKING, List, Union, Optional, Tuple, Dict
import numpy as np
from shapely.geometry import Polygon

import networkx as nx

from powerowl.graph.constants import MODEL_NODE
from powerowl.graph.enums import EdgeType, Layers
from powerowl.graph.enums.layer_matching_strategy import LayerMatchingStrategy
from powerowl.graph.enums.plot_grouping_mode import PlotGroupingMode
from powerowl.graph.model_edge import ModelEdge
from powerowl.graph.model_node import ModelNode
from powerowl.layers.facilities import Facility
from powerowl.layers.network.network_entity import NetworkEntity
from powerowl.layers.network.interface import Interface
from powerowl.layers.network.link import Link
from powerowl.performance.timing import Timing

if TYPE_CHECKING:
    from powerowl.graph import MultiLayerGraph, GraphLayer


class GraphLayout:
    def __init__(self, multi_layer_graph: 'MultiLayerGraph'):
        self.mlg: 'MultiLayerGraph' = multi_layer_graph
        self._config = {
            "node_grouping": PlotGroupingMode.PER_LAYER,
            "edge_grouping": PlotGroupingMode.PER_LAYER,
            "hide_network_link_nodes": False,
            "hide_network_interface_nodes": False
        }
        self._temporary_edges = set()

    def layout(self):
        """
                Calculates a layout for all layers based on the lowest layer
                :return:
                """
        layer: 'GraphLayer'

        for i, layer in enumerate(self.mlg.get_layers(include_sub_layers=False)):
            with Timing(layer.name):
                graph = layer.get_layer_graph(include_sub_layers=True)
                model_nodes = nx.get_node_attributes(graph, MODEL_NODE)
                positions = {node_id: model_node.get_position_2d() for node_id, model_node in model_nodes.items()
                             if model_node.get_position_2d(create_default_position=False) is not None}

                if len(positions) == 0 and len(graph.nodes) > 0:
                    if not layer.has_lower_layer():
                        # Derive a random layout
                        warnings.warn("Lowest layer has no layout! Using standard graph algorithm")
                        with Timing("Graphviz Random").as_sum_timing():
                            pos = nx.random_layout(graph)
                            #pos = nx.nx_agraph.graphviz_layout(graph)
                        self._apply_3d_pos(pos)
                        continue

                    # Derive Layout based on lower layer(s)
                    pos = self._derive_positions(layer, layer_graph=graph)
                    self._apply_3d_pos(pos)
                elif len(positions) != len(graph.nodes):
                    pos2d = {}
                    fixed = []
                    for n in graph.nodes.keys():
                        model_node = self.mlg.get_model_node(n)
                        pos = model_node.get_position_2d(create_default_position=False)
                        if pos is not None:
                            pos2d[n] = pos
                            fixed.append(n)
                            continue
                        # Get position of nearest positioned node
                        for edge in nx.algorithms.traversal.bfs_edges(graph, n):
                            pos = self.mlg.get_model_node(edge[1]).get_position_2d(create_default_position=False)
                            if pos is not None:
                                pos2d[n] = pos
                                break
                    self._apply_3d_pos(pos2d)
                    pos3d = {n: self.mlg.get_model_node(n).get_position_3d() for n in graph.nodes.keys() if n in pos2d}

                    pos3d = self.optimized_spring_layout(graph, pos3d, fixed)
                    #pos3d = self.nx_spring_layout(graph, pos3d, fixed)
                    # pos3d = self.pygraphviz_layout(graph, pos3d, fixed)
                    #self._apply_3d_pos(layer, pos3d, layer.z_offset)
                    self._apply_3d_pos(pos3d)
                else:
                    # Add z-coordinate
                    self._apply_3d_pos(positions)

    def optimized_spring_layout(self, graph: nx.Graph, pos3d: Dict, fixed: List) -> Dict[str, Tuple]:
        """
        Fixes the position of nodes with only two neighbors to simplify the spring layout algorithm.
        Especially useful for network graphs, as links and interfaces can be positioned.
        For all nodes that cannot be positioned, the spring layout is applied
        """
        def get_neighbor_count(_node) -> int:
            return len(list(graph.neighbors(_node)))
        
        def get_fixed_node(_node, _exclude: List) -> Optional[Tuple[int, int]]:
            """
            Follow the node's edges until we find a node with fixed position.
            Does return None when no fixed node can be reached or an unfixed node is found with not exactly 2 neighbors.
            In case a node is found, a Tuple of the fixed node's ID and the hop count is returned.
            """
            if _node in fixed:
                return _node, 0
            if get_neighbor_count(_node) != 2:
                return None
            _neighbors = list(graph.neighbors(_node))
            _n_left = _neighbors[0]
            _n_right = _neighbors[1]
            for _n in [_n_left, _n_right]:
                if _n not in _exclude:
                    _f = get_fixed_node(_n, _exclude + [_node])
                    if _f is not None:
                        return _f[0], _f[1] + 1
            return None

        with Timing("Optimized layout").as_sum_timing():
            unfixable = []
            for n in graph:
                if n in fixed:
                    continue
                if get_neighbor_count(n) != 2:
                    unfixable.append(n)
                    continue
                neighbors = list(graph.neighbors(n))
                n_left = neighbors[0]
                n_right = neighbors[1]
                fixed_left = get_fixed_node(n_left, [n])
                if fixed_left is None:
                    unfixable.append(n)
                    continue
                fixed_right = get_fixed_node(n_right, [n])
                if fixed_right is None:
                    unfixable.append(n)
                    continue
                # Calculate position
                ## Create vector
                fixed_left, hop_left = fixed_left
                fixed_right, hop_right = fixed_right
                hop_left += 1
                hop_right += 1
                pos_left = pos3d[fixed_left]
                pos_right = pos3d[fixed_right]
                ## Vector left -> right
                vector = np.array(pos_right) - np.array(pos_left)
                ## Derive new vector length
                hop_sum = hop_left + hop_right
                v_len = np.linalg.norm(vector)
                hop_len = 1 / hop_sum
                pos_scale = hop_left * hop_len
                pos_node = np.array(pos_left) + (vector * pos_scale)
                fixed.append(n)
                pos3d[n] = pos_node.tolist()

        if len(unfixable) > 0:
            return self.nx_spring_layout(graph, pos3d=pos3d, fixed=fixed)
        return pos3d


    def pygraphviz_layout(self, graph, pos3d, fixed) -> Dict[str, Tuple]:
        import pygraphviz
        # Create AGraph
        a_graph = pygraphviz.AGraph(name=graph.name, strict=True, directed=False)
        a_graph.graph_attr.update(graph.graph.get("graph", {}))
        a_graph.node_attr.update(graph.graph.get("node", {}))
        a_graph.edge_attr.update(graph.graph.get("edge", {}))
        a_graph.graph_attr.update(
            (k, v) for k, v in graph.graph.items() if k not in ("graph", "node", "edge")
        )

        print(f"Got {len(graph.nodes)} nodes")
        print(f"  {len(pos3d)} nodes have 3D position")
        print(f"  {len(fixed)} nodes have fixed position")

        # Add Nodes
        for n in graph.nodes.keys():
            a_graph.add_node(n)
            a_node = a_graph.get_node(n)
            if n in pos3d:
                p = pos3d[n]
                a_node_pos3d = f"{p[0]},{p[1]},{p[2]}"
                if n in fixed:
                    a_node_pos3d = f"{a_node_pos3d}!"
                    a_node.attr["pin"] = "true"
                a_node.attr["pos"] = a_node_pos3d
        print(a_graph)
        # Add Edges
        for u, v in graph.edges():
            a_graph.add_edge(u, v)

        # Apply layout
        with Timing("SFDP").as_sum_timing():
            a_graph.layout(prog="sfdp", args="-Gdim=3")

        # Copy positions
        node_pos = {}
        for n in graph:
            a_node = a_graph.get_node(n)
            pos = a_node.attr["pos"].replace("!", "").split(",")
            node_pos[n] = tuple(float(x) for x in pos)
        return node_pos

    def nx_spring_layout(self, graph, pos3d, fixed):
        with Timing("Spring Layout").as_sum_timing():
            pos3d = nx.spring_layout(graph, pos=pos3d, fixed=fixed, seed=0,
                                     dim=3, k=0.001, iterations=100)
            pos3d = self._avoid_node_overlapping(pos3d)
        return pos3d

    def plotly_figure(self):
        import plotly.graph_objects as go
        z_labels = {}
        traces = []
        default_trace = {
            "x": [],  # Node X Coordinates
            "y": [],  # Node Y Coordinates
            "z": [],  # Node Z Coordinates
            "customdata": [],   # Data to show in node popups
            "labels": [],  # Node Labels
            "colors": [],  # Node Colors
            "xe": [],  # Edge X Coordinates (X-Start, X-End, None)
            "ye": [],  # Edge Y Coordinates (Y-Start, Y-End, None)
            "ze": [],  # Edge Z Coordinates (Z-Start, Z-End, None)
            "customdata-edges": [],   # Data to show in edge popups (Three times!)
            "e_labels": [],  # Edge Labels (Three times!)
            "e_colors": []  # Edge Colors (Three times!)
        }
        sub_traces = {}

        for layer in self.mlg.get_layers(include_sub_layers=True):
            layer_name = layer.name
            # Nodes
            for model_node in self.mlg.get_nodes_at_layers({layer},
                                                           layer_matching_strategy=LayerMatchingStrategy.EXACT):
                z_labels[layer.z] = layer_name
                if not self._is_node_visible(model_node):
                    self._extract_node_temporarily(model_node)
                    continue
                trace_name = self._get_trace_name(model_node)
                if trace_name not in sub_traces:
                    sub_traces[trace_name] = copy.deepcopy(default_trace)
                t = sub_traces[trace_name]
                try:
                    t["x"].append(model_node.get_position_3d("x"))
                    t["y"].append(model_node.get_position_3d("y"))
                    t["z"].append(model_node.get_position_3d("z"))
                    t["labels"].append(model_node.get("label", model_node.name))
                    t["colors"].append(model_node.get_color())
                    t["customdata"].append([
                        # Name
                        model_node.name,
                        # Type
                        model_node.__class__.__name__,
                        # UID
                        model_node.uid,
                        # Layer Name
                        model_node.layer.name,
                        # Facility Name
                        model_node.get_facility().name if model_node.get_facility() is not None else "None",
                        # Node-specific HTML
                        "<br />".join(model_node.get_description_lines())
                    ])
                except Exception as e:
                    print(model_node)
                    raise e
            # Edges
            for model_edge in self.mlg.get_edges_at_layers({layer},
                                                           layer_matching_strategy=LayerMatchingStrategy.EXACT):
                if not self._is_edge_visible(model_edge):
                    continue
                trace_name = self._get_trace_name(model_edge)
                if trace_name not in sub_traces:
                    sub_traces[trace_name] = copy.deepcopy(default_trace)
                    z_labels[layer.z] = layer_name
                t = sub_traces[trace_name]
                edge_label = model_edge.edge_type.value
                if model_edge.edge_type not in [EdgeType.PHYSICAL_POWER, EdgeType.NETWORK_LINK]:
                    t["e_colors"].extend([model_edge.edge_type.get_color()] * 3)
                    t["e_labels"].extend([edge_label] * 3)
                else:
                    t["e_colors"].extend([model_edge.edge_type.get_color()] * 3)
                    t["e_labels"].extend([edge_label] * 3)
                node_a = model_edge.node_a
                node_b = model_edge.node_b
                t["xe"].extend([node_a.pos3d.x, node_b.pos3d.x, None])
                t["ye"].extend([node_a.pos3d.y, node_b.pos3d.y, None])
                t["ze"].extend([node_a.pos3d.z, node_b.pos3d.z, None])
                t["customdata-edges"].extend([[
                    edge_label,
                    node_a.name,
                    node_b.name,
                    ""
                ]] * 3)

        # Inter Layer Edges
        for il_edge in self.mlg.get_inter_layer_edges(layer_matching_strategy=LayerMatchingStrategy.EXACT):
            if not self._is_edge_visible(il_edge):
                continue
            trace_name = self._get_trace_name(il_edge)
            if trace_name not in sub_traces:
                sub_traces[trace_name] = copy.deepcopy(default_trace)
            t = sub_traces[trace_name]
            t["e_colors"].extend([il_edge.edge_type.get_color()] * 3)
            t["e_labels"].extend([il_edge.edge_type.value] * 3)
            node_a = il_edge.node_a
            node_b = il_edge.node_b
            t["xe"].extend([node_a.pos3d.x, node_b.pos3d.x, None])
            t["ye"].extend([node_a.pos3d.y, node_b.pos3d.y, None])
            t["ze"].extend([node_a.pos3d.z, node_b.pos3d.z, None])
            t["customdata-edges"].extend([[
                il_edge.edge_type.value,
                node_a.name,
                node_b.name,
                ""
            ]] * 3)

        for name, t in sub_traces.items():
            if len(t["x"]) > 0:
                traces.append(go.Scatter3d(name=f"{name}", x=t["x"], y=t["y"], z=t["z"],
                                           legendgroup=name,
                                           mode="markers+text", text=t["labels"],
                                           customdata=t["customdata"],
                                           marker={
                                               "color": t["colors"],
                                               "size": 4,
                                               "line": {
                                                   "width": 0.4,
                                                   "color": "DarkSlateGrey"
                                               }
                                           },
                                           hovertemplate='<br />'.join([
                                                "Name:   %{customdata[0]}",
                                                "Type:   %{customdata[1]}",
                                                "UID:    %{customdata[2]}",
                                                "Layer:  %{customdata[3]}",
                                                "Facility: %{customdata[4]}",
                                                "%{customdata[5]}",
                                           ])+"<extra></extra>"))
            if len(t["xe"]) > 0:
                traces.append(go.Scatter3d(name=f"{name}",
                                           x=t["xe"],
                                           y=t["ye"],
                                           z=t["ze"],
                                           legendgroup=name,
                                           customdata=t["customdata-edges"],
                                           mode="lines",
                                           # text=t["e_labels"],
                                           line={"color": t["e_colors"]},
                                           hovertemplate="<br />".join([
                                                "Type:   %{customdata[0]}",
                                                "From:   %{customdata[1]}",
                                                "To:     %{customdata[2]}",
                                                "%{customdata[3]}"
                                           ])+"<extra></extra>"))

        keys = list(z_labels.keys())
        layout = go.Layout(
            scene={
                "xaxis": {
                    "title": {
                        "text": "Position X"
                    }
                },
                "yaxis": {
                    "title": {
                        "text": "Position Y"
                    }
                },
                "zaxis": {
                    "title": {
                        "text": "Layer"
                    },
                    "tickmode": "array",
                    "tickvals": keys,
                    "ticktext": [z_labels[key] for key in keys]
                }
            }
        )
        fig = go.Figure(data=traces, layout=layout)
        return fig

    def _apply_3d_pos(self, pos):
        for n, pos in pos.items():
            model_node = self.mlg.get_model_node(n)
            model_node.set_position_2d(pos[0], pos[1])
            if len(pos) == 3:
                model_node.z_offset = pos[2] - model_node.layer.z

    def _derive_positions(self, layer: 'GraphLayer', layer_graph: nx.Graph):
        graph = layer_graph
        x = 0
        y = 5
        with Timing("NX Random").as_sum_timing():
            pos = nx.random_layout(graph)
        # Calculate base layout
        for n, node in graph.nodes.items():
            # Attempt to derive position
            neighbors = self.mlg.get_inter_layer_neighbors(n)
            positions = []
            for neighbor in neighbors:
                pos_n = neighbor.get_position_2d(create_default_position=False)
                if pos_n is not None:
                    positions.append(pos_n)
            if len(positions) == 0:
                # warnings.warn(f"No derivable position for {n}@{layer}")
                pos[n] = (x, y)
                x += 1
                y += 1
                continue
            if len(positions) == 1:
                c = positions[0]
            elif len(positions) == 2:
                c = ((positions[0][0] + positions[1][0]) / 2, (positions[0][1] + positions[1][1]) / 2)
            else:
                with Timing("Polygon").as_sum_timing():
                    polygon = Polygon(positions)
                    polygon = polygon.convex_hull
                    c = polygon.centroid.coords[0]
            pos[n] = c
        # Avoid node overlapping
        return self._avoid_node_overlapping(pos)

    @staticmethod
    def _avoid_node_overlapping(pos, spacing: float = 0.01):
        if len(pos) == 0:
            return pos

        pos3d = pos
        is_3d = True
        if len(pos[list(pos.keys())[0]]) == 2:
            pos3d = {n: (pos[0], pos[1], 0) for n, pos in pos3d.items()}
            is_3d = False

        with Timing("Overlap Avoidance").as_sum_timing():
            node_groups = {}
            for n, pos in pos3d.items():
                node_groups.setdefault(f"{pos[0]}#{pos[1]}#{pos[2]}", []).append(n)
            overlapping_nodes = [nodes for nodes in node_groups.values() if len(nodes) > 0]
            for node_set in overlapping_nodes:
                c = pos3d[node_set[0]]
                cols = math.ceil(math.sqrt(len(node_set)))
                x_start = c[0] - ((cols - 1)/2) * spacing
                y_start = c[1] - ((cols - 1)/2) * spacing
                z = c[2]
                y = y_start
                col = 0
                for n in node_set:
                    x = x_start + col * spacing
                    pos3d[n] = (x, y, z)
                    col += 1
                    if col >= cols:
                        col = 0
                    y += spacing

                """pos_fixed = pos2d.copy()
                for n in node_set:
                    del pos_fixed[n]
                pos2d = nx.spring_layout(graph, k=spacing, dim=2, pos=pos2d, fixed=pos_fixed, weight=0)"""
        if not is_3d:
            pos2d = {n: (pos[0], pos[1]) for n, pos in pos3d.items()}
            return pos2d
        return pos3d

    def update_config(self, config):
        self._config.update(config)

    def _get_trace_name(self, model_entity: Union[ModelNode, ModelEdge]):
        if isinstance(model_entity, ModelNode):
            node_grouping = self._config.get("node_grouping")

            if node_grouping in [PlotGroupingMode.PER_FACILITY,
                                 PlotGroupingMode.PER_FACILITY_PER_LAYER,
                                 PlotGroupingMode.PER_FACILITY_OR_LAYER]:
                grouper = self.mlg.owl.get_facility(model_entity)
                if grouper is None:
                    grouper = self.mlg.owl.get_ou(model_entity)
                if grouper is None:
                    grouper = model_entity
                grouper_name = grouper.name
                if node_grouping == PlotGroupingMode.PER_FACILITY_PER_LAYER:
                    layer_name = model_entity.layer.name
                    return f"{layer_name} // {grouper_name}"
                if node_grouping == PlotGroupingMode.PER_FACILITY_OR_LAYER:
                    if not isinstance(grouper, Facility):
                        grouper_name = model_entity.layer.name
                return grouper_name

            if node_grouping == PlotGroupingMode.BY_TYPE:
                return model_entity.__class__.__name__

            return model_entity.layer.name

        if isinstance(model_entity, ModelEdge):
            edge_grouping = self._config.get("edge_grouping")

            if edge_grouping == PlotGroupingMode.BY_TYPE:
                return model_entity.edge_type.value

            if edge_grouping == PlotGroupingMode.PER_FACILITY_OR_LAYER:
                facility_a = model_entity.node_a.get_facility()
                facility_b = model_entity.node_b.get_facility()
                if facility_a == facility_b and facility_a is not None:
                    return facility_a.name
                if not model_entity.is_inter_layer_edge(LayerMatchingStrategy.EXACT):
                    return model_entity.node_a.layer.name
                layers = sorted([model_entity.node_a.layer.name, model_entity.node_b.layer.name])
                return f"{layers[0]} -- {layers[1]}"
            name_a = self._get_trace_name(model_entity.node_a)
            name_b = self._get_trace_name(model_entity.node_b)
            if name_a == name_b:
                return name_a
            names = sorted([name_a, name_b])
            return f"{names[0]} -- {names[1]}"

        raise AttributeError("Expecting ModelEdge or ModelNode")

    def _extract_node_temporarily(self, model_node: ModelNode, edge_type: Optional[EdgeType] = None):
        neighbors = self.mlg.get_neighbors(model_node)
        for neighbor_a in neighbors:
            for neighbor_b in neighbors:
                if neighbor_a == neighbor_b:
                    continue
                if self.mlg.has_edge(neighbor_a, neighbor_b):
                    continue
                edge_a = self.mlg.get_edge(model_node, neighbor_a)
                temporary_type = edge_a.edge_type if edge_type is None else edge_type
                temporary_edge = ModelEdge(neighbor_a, neighbor_b, edge_type=temporary_type)
                self.mlg.add_edge(temporary_edge)
                self._temporary_edges.add(temporary_edge)

    def _clear_temporary_edges(self):
        while len(self._temporary_edges) > 0:
            edge = self._temporary_edges.pop()
            if self.mlg.edge_exists(edge):
                self.mlg.remove_edge(edge)

    def _is_edge_visible(self, model_edge: ModelEdge) -> bool:
        return self._is_node_visible(model_edge.node_a) and self._is_node_visible(model_edge.node_b)

    def _is_node_visible(self, model_node: ModelNode) -> bool:
        if isinstance(model_node, NetworkEntity):
            if isinstance(model_node, Interface) and self._config.get("hide_network_interface_nodes"):
                return False
            if isinstance(model_node, Link) and self._config.get("hide_network_link_nodes"):
                return False
        return True
