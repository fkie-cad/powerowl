import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Type, Union

import networkx as nx

from powerowl.exceptions import DerivationError
from powerowl.exceptions.layer_not_found_exception import LayerNotFoundException
from powerowl.exceptions.model_node_not_found_exception import ModelNodeNotFoundException
from powerowl.graph import MultiLayerGraph
from powerowl.graph.enums import Layers
from powerowl.graph.enums.plot_grouping_mode import PlotGroupingMode
from powerowl.graph.model_node import ModelNode
from powerowl.layers.facilities import Facility
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

    def __init__(self, power_grid: Optional['PowerGridModel'] = None, **kwargs):
        """
        Creates a Power Owl Instance.
        """
        self._gid = 0
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
        self.mlg = MultiLayerGraph(owl=self)
        self.power_grid = power_grid
        if power_grid is not None:
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
    def next_global_id() -> int:
        PowerOwl._gid += 1
        return PowerOwl._gid

    def derive(self, derivator_class: Type['Derivator'], **kwargs):
        derivator: 'Derivator' = derivator_class(self, **kwargs)
        derivator.derive()

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

    def derive_ous(self, derivator_classes: Optional[List[Type[Derivator]]] = None):
        if derivator_classes is None:
            from powerowl.derivators.ou.default_ou_derivator import DefaultOUDerivator
            derivator_classes = [DefaultOUDerivator]
        self._multi_derive(derivator_classes)

    def derive_network(self, derivator_classes: Optional[List[Type[Derivator]]] = None):
        if derivator_classes is None:
            from powerowl.derivators.network.default_network_derivator import DefaultNetworkDerivator
            derivator_classes = [DefaultNetworkDerivator]
        self._multi_derive(derivator_classes)

    def get_model_node_of_power_grid_element(self, element: GridElement) -> ModelNode:
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
        power_grid_dict = self.power_grid.to_primitive_dict()
        if not include_facilities:
            return power_grid_dict
        for e_type, elements in self.power_grid.elements.items():
            for e_id, element_dict in elements.items():
                element = self.power_grid.get_element(e_type, e_id)
                node = self.get_model_node_of_power_grid_element(element)
                node_e_type = element.prefix
                node_id = element.index
                facility_id, facility_name = None, None
                facility = self.get_facility(node)
                if facility is not None:
                    facility_id, facility_name = facility.id, facility.name
                power_grid_dict["elements"][node_e_type][node_id]["facility_id"] = facility_id
                power_grid_dict["elements"][node_e_type][node_id]["facility_name"] = facility_name
        return power_grid_dict
