import copy
import enum
import threading
import time
import warnings
from typing import Any, Dict, Type, Tuple, Optional

import pandapower as pp
import pandas as pd

from powerowl.layers.powergrid.elements import *
from powerowl.layers.powergrid.elements.enums.bus_type import BusType
from powerowl.layers.powergrid.elements.enums.connection_type import ConnectionType
from powerowl.layers.powergrid.elements.enums.line_type import LineType
from powerowl.layers.powergrid.elements.enums.static_generator_type import StaticGeneratorType
from powerowl.layers.powergrid.elements.enums.switch_type import SwitchType
from powerowl.layers.powergrid.elements.enums.tap_side import TapSide
from powerowl.layers.powergrid.elements.enums.vector_group import VectorGroup
from powerowl.layers.powergrid.power_grid_model import PowerGridModel
from powerowl.layers.powergrid.power_grid_model_builder import PowerGridModelBuilder
from powerowl.layers.powergrid.values.grid_value import GridValue
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext
from powerowl.layers.powergrid.values.units.scale import Scale
from powerowl.layers.powergrid.values.units.unit import Unit
from powerowl.exceptions import ConversionError


class PandaPowerGridModel(PowerGridModel):
    warn_messages = set()
    enable_warnings: bool = True
    _unassigned_mappings = ["observable"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pp_net: Optional[pp.pandapowerNet] = None
        self._pp_lock = threading.Lock()
        self._last_update = 0

    def simulate(self) -> bool:
        if self._pp_net is None:
            self.prepare_simulator()
        with self._pp_lock:
            sim_net = copy.deepcopy(self._pp_net)
        try:
            pp.runpp(sim_net)
            with self._pp_lock:
                for table in pp.toolbox.pp_elements(other_elements=False, res_elements=True):
                    if table.startswith("res_") and isinstance(sim_net[table], pd.DataFrame):
                        self._pp_net[table] = sim_net[table]
                self._last_update = time.time()
        except Exception as e:
            warnings.warn(f"{e}")
            return False
        return True

    def is_prepared(self) -> bool:
        return self._pp_net is not None

    def prepare_simulator(self):
        self._pp_net = self.to_external()
        for e_type, elements in self.elements.items():
            element: GridElement
            for element in elements.values():
                grid_value: GridValue
                for value_name, grid_value in element.get_grid_values():
                    grid_value.add_on_set_callback(self._on_set_value, as_first=True)
                    grid_value.add_on_before_read_callback(self._on_before_read_value, as_first=True)
                    if grid_value.value_context in [GridValueContext.CONFIGURATION,
                                                    GridValueContext.PROPERTY,
                                                    GridValueContext.GENERIC]:
                        if grid_value.simulator_context is None:
                            continue
                        table, index, column = grid_value.simulator_context
                        _, scale = self.extract_unit_and_scale(column)
                        try:
                            value = grid_value.raw_get_value()
                            value = grid_value.scale.to_scale(value, scale)
                            self._set_pp_value(table, index, column, value)
                        finally:
                            continue

    def get_panda_power_net(self):
        """
        Returns a deepcopy of the internal power grid representation
        """
        if not self.is_prepared():
            self.prepare_simulator()
        return copy.deepcopy(self._pp_net)

    def get_grid_value_by_pandapower_path(self, table: str, index: int, column: str) -> Optional[GridValue]:
        e_type = self.get_element_type_by_table(table)
        element = self.get_element(element_type=e_type, element_id=index)
        for grid_value_info in element.get_grid_values():
            name, grid_value = grid_value_info
            if grid_value.simulator_context == (table, index, column):
                return grid_value
        return None

    def _set_pp_value(self, table, index, column, value):
        with self._pp_lock:
            self._pp_net[table].at[index, column] = value

    def _get_pp_value(self, table, index, column) -> Any:
        with self._pp_lock:
            return self._pp_net[table].at[index, column]

    def _on_before_read_value(self, value: GridValue):
        #if value.last_updated > self._last_update:
        #    return
        if value.value_context in [GridValueContext.MEASUREMENT]:
            if value.simulator_context is None:
                warnings.warn(f"No pandapower context given for {value.name}")
                return
            table, index, column = value.simulator_context
            unit, scale = self.extract_unit_and_scale(column)
            try:
                value.set_value(self._get_pp_value(table, index, column), value_scale=scale)
            except KeyError:
                return

    def _on_set_value(self, value: GridValue, old_value: Any, new_value: Any):
        if old_value == new_value:
            return
        if value.value_context not in [GridValueContext.CONFIGURATION]:
            return
        if value.simulator_context is None:
            warnings.warn(f"No pandapower context given for {value.name}")
            return
        table, index, column = value.simulator_context
        unit, scale = self.extract_unit_and_scale(column)
        if value.value_type in [float, int]:
            new_value = value.scale.to_scale(new_value, scale)
        else:
            new_value = value.value
        self._set_pp_value(table, index, column, new_value)

    def to_external(self) -> Any:
        net = pp.create_empty_network()
        for e_type, elements in self.elements.items():
            for e_id, element in elements.items():
                PandaPowerGridModel._from_wattson_elem(element, net)
        self._add_simulator_context()
        return net

    def from_external(self, external_model: Any):
        net: pp.pandapowerNet = external_model
        builder = self.builder

        # Bus
        PandaPowerGridModel._to_wattson_elem(net, 'bus', Bus, builder)
        # Line
        PandaPowerGridModel._to_wattson_elem(net, 'line', Line, builder)
        # Transformer
        PandaPowerGridModel._to_wattson_elem(net, 'trafo', Transformer, builder)
        # Three Winding Transformer
        PandaPowerGridModel._to_wattson_elem(net, 'trafo3w', None, builder)
        # External Grid
        PandaPowerGridModel._to_wattson_elem(net, 'ext_grid', ExternalGrid, builder)
        # Switch
        PandaPowerGridModel._to_wattson_elem(net, 'switch', Switch, builder)
        # Load
        PandaPowerGridModel._to_wattson_elem(net, 'load', Load, builder)
        # Asymmetric Load
        PandaPowerGridModel._to_wattson_elem(net, 'asymmetric_load', None, builder)
        # Motor
        PandaPowerGridModel._to_wattson_elem(net, 'motor', Motor, builder)
        # Static Generator
        PandaPowerGridModel._to_wattson_elem(net, 'sgen', StaticGenerator, builder)
        # Asymmetric Static Generator
        PandaPowerGridModel._to_wattson_elem(net, 'asymmetric_sgen', None, builder)
        # Generator
        PandaPowerGridModel._to_wattson_elem(net, 'gen', Generator, builder)
        # Shunt
        PandaPowerGridModel._to_wattson_elem(net, 'shunt', Shunt, builder)
        # Impedance
        PandaPowerGridModel._to_wattson_elem(net, 'impedance', Impedance, builder)
        # Ward
        PandaPowerGridModel._to_wattson_elem(net, 'ward', Ward, builder)
        # Extended Ward
        PandaPowerGridModel._to_wattson_elem(net, 'xward', ExtendedWard, builder)
        # DC Line
        PandaPowerGridModel._to_wattson_elem(net, 'dcline', DcLine, builder)
        # Storage
        PandaPowerGridModel._to_wattson_elem(net, 'storage', Storage, builder)
        self._add_simulator_context()

    @staticmethod
    def warn(msg: str):
        if not PandaPowerGridModel.enable_warnings:
            return
        if msg in PandaPowerGridModel.warn_messages:
            return
        PandaPowerGridModel.warn_messages.add(msg)
        print(f"Warning: {msg}")

    @staticmethod
    def _from_wattson_elem(element: GridElement, net: pp.pandapowerNet):
        """
        Adds a pandapower element to the given pandapowerNet representing the given PowerOwl GridElement.
        """
        prefix = element.prefix
        table = element.prefix
        index = element.index
        # Handle non-standard prefixes
        if isinstance(element, ExternalGrid):
            prefix = "ext_grid"
            table = prefix
        elif isinstance(element, ExtendedWard):
            prefix = "xward"
            table = prefix
        elif isinstance(element, Line):
            prefix = "line_from_parameters"
        elif isinstance(element, Transformer):
            prefix = "transformer_from_parameters"
        # Check if element already exists in the network
        if table in net and index in net[table].index:
            return
        # Check for dependencies
        for dependency in element.get_elements_in_attributes():
            PandaPowerGridModel._from_wattson_elem(element=dependency, net=net)

        # Dynamically get method for creating a pandapower element
        pandapower_method_name = f"create_{prefix}"
        try:
            pandapower_method = getattr(pp, pandapower_method_name)
        except AttributeError:
            raise ConversionError(f"Could not create pandapower element for {element.get_identifier()}")
        # Call method
        attributes = {}
        for attribute_name, attribute in element.get_grid_values(GridValueContext.GENERIC):
            PandaPowerGridModel._translate_attribute(attribute_name, attribute, element, net, attributes)
        for attribute_name, attribute in element.get_grid_values(GridValueContext.PROPERTY):
            PandaPowerGridModel._translate_attribute(attribute_name, attribute, element, net, attributes)
        for attribute_name, attribute in element.get_grid_values(GridValueContext.CONFIGURATION):
            PandaPowerGridModel._translate_attribute(attribute_name, attribute, element, net, attributes)
        pandapower_method(net, **attributes)

    @staticmethod
    def get_pandapower_table(element: GridElement):
        table = element.prefix
        if isinstance(element, ExternalGrid):
            table = "ext_grid"
        elif isinstance(element, ExtendedWard):
            table = "xward"
        return table

    @staticmethod
    def get_element_type_by_table(table: str):
        return table.replace("res_", "").replace("_est", "")

    def _add_simulator_context(self):
        for elements in self.elements.values():
            for element in elements.values():
                for attribute_name, attribute in element.get_grid_values():
                    column = PandaPowerGridModel.column_mapping().get(attribute_name)
                    table = self.get_pandapower_table(element)
                    index = element.index
                    value_context = attribute.value_context
                    if attribute.value_simulator_context is not None:
                        value_context = attribute.value_simulator_context
                    if element.prefix == "switch" and attribute_name == "is_closed":
                        value_context = GridValueContext.CONFIGURATION
                    if element.prefix == "storage" and attribute_name == "state_of_charge":
                        value_context = GridValueContext.CONFIGURATION

                    if value_context == GridValueContext.MEASUREMENT:
                        table = f"res_{table}"
                    if value_context == GridValueContext.ESTIMATION:
                        table = f"res_{table}_est"
                    if column is not None:
                        attribute.simulator_context = (table, index, column)

    @staticmethod
    def _to_wattson_elem(net, tbl: str, cls: Type[GridElement], builder: 'PowerGridModelBuilder'):
        if tbl in net and len(net[tbl]) > 0:
            if cls is None:
                raise NotImplementedError(f"No matching class for {tbl} elements")
            attribute_list = cls.get_specifications()
            attributes = {}
            for a in attribute_list:
                attributes[a.name] = a

            cm = PandaPowerGridModel.column_mapping()

            for i, row_dict in net[tbl].to_dict(orient="index").items():
                element_pp_data = row_dict
                config = {}
                config.update(PandaPowerGridModel.handle_element_specials(element_pp_data, cls, builder))
                if cls == Bus:
                    # Check for geodata
                    if "bus_geodata" in net:
                        if i in net.bus_geodata.index:
                            if "coords" in net.bus_geodata.columns:
                                element_pp_data["coords"] = net.bus_geodata.at[i, "coords"]
                            if "x" in net.bus_geodata.columns and "y" in net.bus_geodata.columns:
                                element_pp_data["geodata"] = (net.bus_geodata.at[i, "x"], net.bus_geodata.at[i, "y"])

                cols = list(element_pp_data.keys())

                for name, a in attributes.items():
                    if name in config:
                        continue
                    if a.context not in [GridValueContext.GENERIC,
                                         GridValueContext.PROPERTY,
                                         GridValueContext.CONFIGURATION]:
                        continue
                    col = cm.get(name) if a.pp_column is None else a.pp_column
                    if col is None and name not in PandaPowerGridModel._unassigned_mappings:
                        PandaPowerGridModel.warn(
                            f"Unknown mapping for attribute {name} for {cls.__name__} ({tbl})"
                        )
                        continue
                    if col not in cols:
                        if a.required:
                            PandaPowerGridModel.warn(f"Unknown column {col} for {cls.__name__} ({tbl})")
                        continue
                    grid_value = GridValue.from_specification(a, None)

                    unit, scale = PandaPowerGridModel.extract_unit_and_scale(col)
                    if unit != Unit.NONE and grid_value.unit != unit:
                        PandaPowerGridModel.warn(
                            f"Unit mismatch for {name} // {col} for {cls.__name__} ({grid_value.unit} vs {unit})")
                        continue

                    val = element_pp_data[col]
                    if isinstance(a.value_type, type) and issubclass(a.value_type, enum.Enum):
                        val = PandaPowerGridModel.translate_value_to_enum(cls.prefix, col, val)
                    if name.endswith("bus"):
                        # Get bus
                        bus_id = element_pp_data[col]
                        val = builder.get_element("bus", bus_id)

                    grid_value.set_value(val, None, scale)
                    config[name] = grid_value

                config["index"] = i
                builder.create_elem(cls, **config)

    @staticmethod
    def _translate_attribute(attribute_name: str, attribute: GridValue, element: GridElement, net: pp.pandapowerNet,
                             attributes: dict):
        if attribute_name in PandaPowerGridModel._unassigned_mappings:
            return
        if attribute.value is None:
            # Skip empty values
            return
        v = attribute.value
        # Get target column
        column = PandaPowerGridModel.column_mapping().get(attribute_name)
        if column is None:
            PandaPowerGridModel.warn(f"Could not find suitable pandapower column for {attribute_name} "
                                     f"of {element.get_identifier()}")
            return
        # Transcribe value
        if isinstance(v, GridElement):
            v = v.index
        elif isinstance(v, enum.Enum):
            v = PandaPowerGridModel._translate_enum_to_value(v)
        else:
            # Handle different scales
            value_scale = attribute.scale
            if value_scale != Scale.NONE:
                unit, scale = PandaPowerGridModel.extract_unit_and_scale(column=column)
                v = value_scale.to_scale(v, scale)
        if isinstance(element, Line) and column == "std_type":
            # When we create the line from parameters, we cannot set std_type (for some pandapower reason - introduced with 2.13.X)
            return
        if isinstance(element, Transformer) and column == "vector_group":
            # Another pandapower breaking change - vector_group...
            return
        attributes[column] = v
        # Potentially handle associated attributes
        PandaPowerGridModel._handle_associated_attributes(element, attribute_name, attribute, net, attributes)

    @staticmethod
    def _handle_associated_attributes(element: GridElement, value_name: str, value: GridValue,
                                      net: pp.pandapowerNet, attributes: dict):
        """
        Creates additional attributes that are necessary for pandapower but not for PowerOwl,
        e.g., the element type field for switches.
        """
        if isinstance(element, Switch) and value_name == "element":
            # Create et attribute (element type)
            if isinstance(value.value, Bus):
                attributes["et"] = "b"
            elif isinstance(value.value, (Line, DcLine)):
                attributes["et"] = "l"
            elif isinstance(value.value, Transformer):
                # TODO: Include three winding transformer
                attributes["et"] = "t"
            else:
                PandaPowerGridModel.warn(f"Invalid element associated with {element.get_identifier()}: {value.value}")

    @staticmethod
    def _translate_enum_to_value(value: enum.Enum) -> Any:
        if isinstance(value, BusType):
            return {
                BusType.NONE: None,
                BusType.BUSBAR: "b",
                BusType.NODE: "n",
                BusType.MUFF: "m"
            }[value]
        if isinstance(value, TapSide):
            return {
                TapSide.LV: "lv",
                TapSide.HV: "hv",
                TapSide.NONE: None
            }[value]
        if isinstance(value, VectorGroup):
            return value.value
        if isinstance(value, SwitchType):
            return {
                SwitchType.LOAD_SWITCH: "LS",
                SwitchType.DISCONNECTING_SWITCH: "DS",
                SwitchType.LOAD_BREAK_SWITCH: "LBS",
                SwitchType.CIRCUIT_BREAKER: "CB",
                SwitchType.NONE: None
            }[value]
        if isinstance(value, StaticGeneratorType):
            return {
                StaticGeneratorType.PHOTOVOLTAIC: "PV",
                StaticGeneratorType.WIND: "WP",
                StaticGeneratorType.COMBINED_HEATING_AND_POWER: "CHP",
                StaticGeneratorType.COMBINED_HEATING_AND_POWER_DIESEL: "CHP diesel",
                StaticGeneratorType.FUEL_CELL: "Fuel cell",
                StaticGeneratorType.FUEL_CELL_RESIDENTIAL: "Residential fuel cell",
                StaticGeneratorType.NONE: None
            }[value]
        if isinstance(value, LineType):
            return {
                LineType.NONE: None,
                LineType.OVERHEAD_LINE: "ol",
                LineType.UNDERGROUND_CABLE: "cs"
            }[value]
        if isinstance(value, ConnectionType):
            return {
                ConnectionType.NONE: None,
                ConnectionType.WYE: "wye",
                ConnectionType.DELTA: "delta"
            }[value]
        raise ConversionError(f"Enum of type {value.__class__.__name__} cannot be converted")

    """
    Pandapower to PowerOwl
    """
    @staticmethod
    def handle_element_specials(row_dict, cls: Type[GridElement], builder: 'PowerGridModelBuilder') -> dict:
        e_type = cls.prefix
        config = {}
        if e_type == "switch":
            et = row_dict["et"]
            connected_type = {
                "l": "line",
                "b": "bus",
                "t": "trafo"
            }[et]
            connected_element = builder.get_element(connected_type, row_dict["element"])
            config["element"] = connected_element
        return config

    @staticmethod
    def translate_value_to_enum(element_type: str, column: str, value: str) -> enum.Enum:
        if isinstance(value, str):
            value = value.lower()
        if element_type == "bus":
            if column == "type":
                return {
                    "b": BusType.BUSBAR,
                    "n": BusType.NODE,
                    "m": BusType.MUFF,
                }.get(value, BusType.NONE)
        if element_type == "trafo" and column == "tap_side":
            return {
                "lv": TapSide.LV,
                "hv": TapSide.HV
            }.get(value, TapSide.NONE)
        if element_type == "switch" and column == "type":
            return {
                "ls": SwitchType.LOAD_SWITCH,
                "cb": SwitchType.CIRCUIT_BREAKER,
                "lbs": SwitchType.LOAD_BREAK_SWITCH,
                "ds": SwitchType.DISCONNECTING_SWITCH
            }.get(value, SwitchType.NONE)
        if element_type == "sgen" and column == "type":
            return {
                "pv": StaticGeneratorType.PHOTOVOLTAIC,
                "wp": StaticGeneratorType.WIND,

                "chp": StaticGeneratorType.COMBINED_HEATING_AND_POWER,
                "chp diesel": StaticGeneratorType.COMBINED_HEATING_AND_POWER_DIESEL,

                "residential fuel cell": StaticGeneratorType.FUEL_CELL_RESIDENTIAL,
                "fuel cell": StaticGeneratorType.FUEL_CELL,
                "fc": StaticGeneratorType.FUEL_CELL,
                "rfc": StaticGeneratorType.FUEL_CELL_RESIDENTIAL
            }.get(value, StaticGeneratorType.NONE)
        if element_type in ["load", "sgen"] and column == "type":
            return {
                "wye": ConnectionType.WYE,
                "delta": ConnectionType.DELTA
            }.get(value, ConnectionType.NONE)
        if element_type == "line" and column == "type":
            return {
                "cs": LineType.UNDERGROUND_CABLE,
                "ol": LineType.OVERHEAD_LINE
            }.get(value, LineType.NONE)

        raise ValueError(f"No Enum found for {element_type}.{column} ({value=})")

    @staticmethod
    def extract_unit_and_scale(column: str) -> Tuple[Unit, Scale]:
        indicator = column.split("_")[-1]
        if column in ["r_ohm_per_km", "x_ohm_per_km"]:
            return Unit.OHM, Scale.BASE
        if column == "c_nf_per_km":
            return Unit.FARAD, Scale.BASE
        if indicator == "mw":
            return Unit.WATT, Scale.MEGA
        if indicator == "mva":
            return Unit.VOLT_AMPERE, Scale.MEGA
        if indicator == "mwh":
            return Unit.WATT_HOUR, Scale.MEGA
        if indicator == "kw":
            return Unit.WATT, Scale.KILO
        if indicator == "mvar":
            return Unit.VAR, Scale.MEGA
        if indicator == "kv":
            return Unit.VOLT, Scale.KILO
        if indicator == "ka":
            return Unit.AMPERE, Scale.KILO
        if indicator == "degree":
            return Unit.DEGREE, Scale.NONE
        if indicator == "km":
            return Unit.METER, Scale.KILO
        if indicator == "pu":
            return Unit.PER_UNIT, Scale.NONE
        if indicator == "percent":
            return Unit.PERCENT, Scale.NONE
        return Unit.NONE, Scale.NONE

    @staticmethod
    def column_mapping() -> Dict:
        return {
            "name": "name",
            "type_name": "std_type",
            "generator_type": "type",
            "type": "type",
            "connection_type": "type",
            "element": "element",
            "zone": "zone",
            "position": "coords",
            "geo_position": "geodata",

            "nominal_power": "sn_mva",

            "active_power": "p_mw",
            "active_power_loss": "pl_mw",
            "target_active_power": "p_mw",
            "maximum_active_power": "max_p_mw",
            "minimum_active_power": "min_p_mw",
            "active_power_from": "p_from_mw",
            "active_power_to": "p_to_mw",
            "active_power_hv": "p_hv_mw",
            "active_power_lv": "p_lv_mw",

            "target_reactive_power": "q_mvar",
            "reactive_power": "q_mvar",
            "reactive_power_loss": "ql_mvar",
            "maximum_reactive_power": "max_q_mvar",
            "minimum_reactive_power": "min_q_mvar",
            "reactive_power_from": "q_from_mvar",
            "reactive_power_to": "q_to_mvar",
            "reactive_power_hv": "q_hv_mvar",
            "reactive_power_lv": "q_lv_mvar",

            "voltage": "vm_pu",
            "target_voltage": "vm_pu",
            "voltage_niveau": "vn_kv",
            "voltage_angle": "va_degree",
            "voltage_angle_from": "va_from_degree",
            "voltage_angle_to": "va_to_degree",
            "voltage_angle_hv": "va_hv_degree",
            "voltage_angle_lv": "va_lv_degree",
            "target_voltage_angle": "va_degree",
            "minimum_voltage": "min_vm_pu",
            "maximum_voltage": "max_vm_pu",
            "voltage_from": "vm_from_pu",
            "voltage_to": "vm_to_pu",
            "voltage_hv": "vm_hv_pu",
            "voltage_lv": "vm_lv_pu",
            "voltage_niveau_hv": "vn_hv_kv",
            "voltage_niveau_lv": "vn_lv_kv",

            "current": "i_ka",
            "current_from": "i_from_ka",
            "current_to": "i_to_ka",
            "current_lv": "i_lv_ka",
            "current_hv": "i_hv_ka",
            "maximum_current": "max_i_ka",

            # Line properties
            "capacitance_per_km": "c_nf_per_km",
            "reactance_per_km": "x_ohm_per_km",
            "resistance_per_km": "r_ohm_per_km",
            "zero_sequence_resistance_per_km": "r0_ohm_per_km",
            "zero_sequence_reactance_per_km": "x0_ohm_per_km",
            "zero_sequence_capacitance_per_km": "c0_nf_per_km",

            # Transformer properties
            "parallel": "parallel",
            "vector_group": "vector_group",
            "phase_shift": "shift_degree",
            "short_circuit_voltage": "vk_percent",
            "short_circuit_voltage_real": "vkr_percent",
            "iron_power_loss": "pfe_kw",
            "open_loop_loss": "i0_percent",
            "zero_sequence_short_circuit_voltage": "vk0_percent",
            "zero_sequence_short_circuit_voltage_real": "vkr0_percent",
            "magnetizing_ration": "mag0_percent",
            "zero_sequence_magnetizing_ration": "mag0_rx",
            "zero_sequence_short_circuit_impedance": "si0_hv_partial",
            "tap_neutral": "tap_neutral",
            "tap_minimum": "tap_min",
            "tap_maximum": "tap_max",
            "tap_position": "tap_pos",
            "tap_phase_shifter": "tap_phase_shifter",
            "tap_side": "tap_side",
            "voltage_tap_step": "tap_step_percent",
            "angle_tap_step": "tap_step_degree",
            "maximum_load": "max_loading_percent",

            # Load
            "constant_current": "const_i_percent",
            "constant_impedance": "const_z_percent",

            # Storage
            "maximum_charge": "max_e_mwh",
            "minimum_charge": "min_e_mwh",
            "state_of_charge": "soc_percent",

            # External Grid
            "maximum_short_circuit_power_provision": "s_sc_max_mva",
            "minimum_short_circuit_power_provision": "s_sc_min_mva",
            "maximum_short_circuit_impedance_rx_ratio": "rx_max",
            "minimum_short_circuit_impedance_rx_ratio": "rx_min",
            "maximum_zero_sequence_rx_ratio": "r0x0_max",
            "maximum_zero_sequence_x0x_ratio": "x0x_max",

            "in_service": "in_service",
            "controllable": "controllable",
            "loading": "loading_percent",
            "cos_phi": "cos_phi",
            "scaling": "scaling",
            "closed": "closed",
            "is_closed": "closed",

            "length": "length_km",

            "from_bus": "from_bus",
            "to_bus": "to_bus",
            "bus": "bus",
            "lv_bus": "lv_bus",
            "hv_bus": "hv_bus",
        }

    @staticmethod
    def reverse_column_mapping():
        cm = PandaPowerGridModel.column_mapping()
        rcm = {}
        for key, value in cm.items():
            rcm.setdefault(value, []).append(key)
        return rcm
