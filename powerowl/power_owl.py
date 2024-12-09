import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Type, Union, Dict

import networkx as nx

from powerowl.exceptions import DerivationError
from powerowl.exceptions.layer_not_found_exception import LayerNotFoundException
from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.exceptions.reiterate_derivation_exception import ReiterateDerivationException
from powerowl.graph import MultiLayerGraph
from powerowl.graph.enums import Layers
from powerowl.graph.enums.plot_grouping_mode import PlotGroupingMode
from powerowl.graph.model_edge import ModelEdge
from powerowl.graph.model_node import ModelNode
from powerowl.layers.facilities import Facility
from powerowl.layers.network.router import Router
from powerowl.layers.network.switch import Switch
from powerowl.layers.network.rtu import RTU
from powerowl.layers.network.mtu import MTU
from powerowl.layers.ou import OrganizationalUnit
from powerowl.layers.powergrid import PowerGridModel
from powerowl.layers.powergrid.elements import GridElement
from powerowl.performance.timing import Timing

if TYPE_CHECKING:
    from powerowl.derivators.derivator import Derivator


class PowerOwl:
    """
    Allows to derive a power grid infrastructure with facilities, network topologies, devices and configuration,
    safety devices and their relations from a given technical power grid representation.
    The whole model is represented by a multi-layered-graph (MLG) that represents entities and their relations
    on different layers (e.g., the power grid layer represents all power grid assets, while multiple network layers
    contain network equipment).
    The derivation process is implemented by using different Derivators, that are specialized on deriving a
    specific aspect of the final model.
    The PowerOwl model can be saved to and restored from a textual representation and can further be exported
    to a scenario configuration to be used with simulation or emulation tools.
    """
    _gid: int = 0
    _typed_ids: Dict[Type, int] = {}

    def __init__(self, power_grid: Optional['PowerGridModel'] = None, **kwargs):
        """
        Creates a Power Owl Instance.
        """
        self._gid = 0
        self.mlg: Optional[MultiLayerGraph] = None
        self.power_grid: Optional['PowerGridModel'] = power_grid
        self._config = {
            "show_warnings": False,
            "draw_node_grouping": PlotGroupingMode.PER_FACILITY_OR_LAYER,
            "draw_edge_grouping": PlotGroupingMode.PER_FACILITY_OR_LAYER,
            "draw_hide_network_link_nodes": False,
            "draw_hide_network_interface_nodes": False
        }
        self._data = {}
        if "enable_timing" in kwargs:
            Timing.enabled = kwargs.get("enable_timing", True)
        self._config.update(kwargs)
        self._reset()

    def _reset(self):
        self.reset_global_id()
        PowerOwl._typed_ids = {}
        self.mlg = MultiLayerGraph(owl=self)
        if self.power_grid is not None:
            self.mlg.add_layer(Layers.POWER_GRID, from_graph=self.power_grid.to_nx_graph())
            self.mlg.add_layer_object(Layers.POWER_GRID, self.power_grid)

    @property
    def enable_timing(self) -> bool:
        return Timing.enabled

    @enable_timing.setter
    def enable_timing(self, value: bool):
        Timing.enabled = value

    def configure(self, **kwargs):
        self._config.update(kwargs)

    def set_data(self, context, key, value):
        self._data.setdefault(context, {})[key] = value

    def append_data(self, context, key, value):
        try:
            self._data.setdefault(context, {}).setdefault(key, []).append(value)
        except Exception:
            raise KeyError(f"{key} is not of type list for context {context}")

    def extend_data(self, context, key, values):
        try:
            self._data.setdefault(context, {}).setdefault(key, []).extend(values)
        except Exception:
            raise KeyError(f"{key} is not of type list for context {context}")

    def get_data(self, context, key = None, default_value=None):
        if key is None:
            return self._data.get(context, {})
        return self._data.get(context, {}).get(key, default_value)

    def encode(self, include_layer_objects: bool = True) -> dict:
        """
        Creates a JSON-serializable dict representation of this power owl instance.
        """
        return {
            "multi_layer_graph": self.mlg.encode(include_layer_objects=include_layer_objects)
        }

    def decode(self, owl_dict: dict):
        """
        Restores a PowerOwl instance from a dictionary representation created by the encode method.
        """
        graph_dict = owl_dict["multi_layer_graph"]
        self.mlg.decode(graph_dict)

    def save_to_file(self, output: Path, indent: Optional[int] = None):
        """
        Saves a JSON representation of this PowerOwl instance to the given file.
        Optionally allows to specify a nesting-indentation.
        """
        encoded_owl = self.encode()
        with output.open("w") as f:
            json.dump(encoded_owl, f, indent=indent)
        return output

    def load_from_file(self, source_file: Path):
        with source_file.open("r") as f:
            encoded_owl = json.load(fp=f)
        self.decode(encoded_owl)

    def export_scenario(self,
                        target_folder: Path,
                        *,
                        split_data_points: bool = True,
                        export_mac_addresses: bool = True,
                        create_extension_basics: bool = True):
        """
        Creates a scenario configuration from this PowerOwl instance.
        The result is written to the given folder, which will be created if not existent.
        """
        from powerowl.export.scenario_exporter import ScenarioExporter
        exporter = ScenarioExporter(self,
                                    clear_folder=True,
                                    split_data_points=split_data_points,
                                    fixed_mac_addresses=export_mac_addresses,
                                    create_extension_basics=create_extension_basics)
        return exporter.export(target_folder)

    @staticmethod
    def next_global_id(model_element: Optional[Union[ModelNode, ModelEdge]] = None) -> int:
        PowerOwl._gid += 1
        return PowerOwl._gid

    @staticmethod
    def next_typed_id(cls: Type) -> int:
        PowerOwl._typed_ids.setdefault(cls, 0)
        PowerOwl._typed_ids[cls] += 1
        return PowerOwl._typed_ids[cls]

    @staticmethod
    def reset_global_id():
        PowerOwl._gid = 0

    def derive(self, derivator_class: Type['Derivator'], **kwargs):
        namesets = kwargs.pop('namesets', [])

        iterate = True
        iteration = 0
        max_iterations = 10
        while iterate:
            use_stable_ids = self._config.get("use_stable_ids", True)
            iterate = False
            iteration += 1
            if iteration > max_iterations:
                print(f"No more iterations available. Giving up")
                return
            try:
                derivator: 'Derivator' = derivator_class(self, **kwargs)
                derivator.derive()
                if use_stable_ids:
                    print("Applying stable IDs")
                    self._apply_stable_ids()
            except ReiterateDerivationException as e:
                print(f"Reiteration requested in iteration {iteration}")
                config = e.updated_configuration
                grid = e.power_grid_model
                kwargs["config"] = config
                self.power_grid = grid
                for nameset in namesets:
                    nameset.reset()
                self._reset()
                iterate = True

    def layout(self):
        return self.mlg.layout()

    def draw(self):
        draw_config = {}
        for key, value in self._config.items():
            if key.startswith("draw_"):
                draw_config[key[5:]] = value
        return self.mlg.draw(**draw_config)

    def _multi_derive(self, derivator_classes: List):
        for derivator_class in derivator_classes:
            derivator = derivator_class(self)
            try:
                derivator.derive()
            except DerivationError as e:
                raise e

    def derive_ous(self, derivator_classes: Optional[List[Type['Derivator']]] = None):
        if derivator_classes is None:
            from powerowl.derivators.ou.default_ou_derivator import DefaultOUDerivator
            derivator_classes = [DefaultOUDerivator]
        self._multi_derive(derivator_classes)

    def derive_network(self, derivator_classes: Optional[List[Type['Derivator']]] = None):
        if derivator_classes is None:
            from powerowl.derivators.network.default_network_derivator import DefaultNetworkDerivator
            derivator_classes = [DefaultNetworkDerivator]
        self._multi_derive(derivator_classes)

    def get_model_node_of_power_grid_element(self, element: GridElement) -> ModelNode:
        # Try to get node by identifier
        try:
            node = self.mlg.get_model_node(element.get_identifier())
            if node.get_grid_element() == element:
                return node
        except KeyError:
            pass

        # Fallback: Full search
        for model_node in self.mlg.get_nodes_at_layers({self.mlg.get_layer(Layers.POWER_GRID)}):
            if model_node.get_grid_element() == element:
                return model_node
        raise ModelNodeNotFoundException(f"No ModelNode for grid element {element.get_identifier()=}")

    def get_ou(self, node: Union[str, ModelNode]) -> Optional[OrganizationalUnit]:
        model_node = self.mlg.get_model_node(node)
        try:
            ous = self.mlg.get_nodes_at_layers({self.mlg.get_layer(Layers.OUs)})
        except LayerNotFoundException:
            return None
        paths = {}
        for ou in ous:
            try:
                path = nx.shortest_path(self.mlg.graph, model_node.uid, ou.uid)
                paths[ou] = path
            except nx.exception.NetworkXNoPath:
                continue
        shortest_path = None
        closest_ou = None
        for ou, path in paths.items():
            if shortest_path is None:
                shortest_path = path
                closest_ou = ou
            elif len(path) < len(shortest_path):
                shortest_path = path
                closest_ou = ou
        return closest_ou

    def get_facility(self, node: Union[str, 'ModelNode']) -> Optional[Facility]:
        model_node = self.mlg.get_model_node(node)
        if not self.mlg.has_layer(Layers.FACILITIES):
            return None
        layer = self.mlg.get_layer(Layers.FACILITIES)
        if isinstance(model_node, Facility):
            return model_node
        if model_node.layer.get_main_layer().z > layer.z:
            return None
        neighbors = self.mlg.get_inter_layer_neighbors(model_node, other_layers={layer})
        for neighbor in neighbors:
            if isinstance(neighbor, Facility):
                return neighbor
        return None

    def get_power_grid_dict(self, *, include_facilities: bool = True) -> dict:
        if not include_facilities:
            return self.power_grid.to_primitive_dict()
        # Add Facilities
        for e_type, elements in self.power_grid.elements.items():
            for e_id, element_dict in elements.items():
                element = self.power_grid.get_element(e_type, e_id)
                node = self.get_model_node_of_power_grid_element(element)
                facility = self.get_facility(node)
                if facility is not None:
                    element.set_data("facility_id", facility.id)
                    element.set_data("facility_name", facility.name)
        return self.power_grid.to_primitive_dict()

    def _apply_stable_ids(self, step: int = 100, minimum_gap: int = 50):
        """
        Updates all node IDs to be more stable between different iterations
        """
        priorities = {
            ModelNode: -1,
            Facility: 1,
            MTU: 100,
            RTU: 50,
            Switch: 40,
            Router: 45
        }
        node_types = sorted(self._typed_ids.keys(), key=lambda cls: priorities.get(cls, 0), reverse=True)
        id_map = {}

        def apply_step(_id) -> int:
            r = _id % step
            _id = _id + (step - r)
            if step - r < minimum_gap:
                _id += step
            return _id

        remapped_id: int = 0
        step_required = False
        for node_type in node_types:
            if step_required:
                pre_step = remapped_id
                remapped_id = apply_step(remapped_id)
            step_required = False
            for node in self.mlg.get_nodes_by_class(node_type):
                remapped_id += 1
                id_map[node.uid] = remapped_id
                step_required = True

        pre_step = remapped_id
        remapped_id = apply_step(remapped_id)
        for edge in self.mlg.get_edges():
            remapped_id += 1
            edge.set_id(remapped_id)

        self.mlg.relabel_nodes(id_map)

