import copy
import datetime
import enum
import math
import threading
import time
import traceback
import warnings
from typing import Any, Dict, Type, Tuple, Optional, Callable, List

import numpy as np
import pandapower as pp
import pandas
import pandas as pd

from powerowl.layers.powergrid.elements import *
from powerowl.layers.powergrid.elements.enums.bus_type import BusType
from powerowl.layers.powergrid.elements.enums.connection_type import ConnectionType
from powerowl.layers.powergrid.elements.enums.generator_type import GeneratorType
from powerowl.layers.powergrid.elements.enums.line_type import LineType
from powerowl.layers.powergrid.elements.enums.reactive_power_mode import ReactivePowerMode
from powerowl.layers.powergrid.elements.enums.static_generator_type import StaticGeneratorType
from powerowl.layers.powergrid.elements.enums.storage_type import StorageType
from powerowl.layers.powergrid.elements.enums.switch_type import SwitchType
from powerowl.layers.powergrid.elements.enums.tap_side import TapSide
from powerowl.layers.powergrid.elements.enums.vector_group import VectorGroup
from powerowl.layers.powergrid.elements.grid_asset import GridAsset
from powerowl.layers.powergrid.power_grid_model import PowerGridModel
from powerowl.layers.powergrid.power_grid_model_builder import PowerGridModelBuilder
from powerowl.layers.powergrid.values.grid_value import GridValue
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext
from powerowl.layers.powergrid.values.grid_value_type import Step
from powerowl.layers.powergrid.values.units.scale import Scale
from powerowl.layers.powergrid.values.units.unit import Unit
from powerowl.exceptions import ConversionError
from powerowl.simulators.pandapower.processors.processor import Processor
from powerowl.simulators.pandapower.processors.processor_return_action import ProcessorReturnAction
from powerowl.simulators.pandapower.processors.protection_processor import ProtectionProcessor
from powerowl.simulators.pandapower.processors.storage_processor import StorageProcessor


class PandaPowerGridModel(PowerGridModel):
    warn_messages = set()
    enable_warnings: bool = True
    _unassigned_mappings = ["observable"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pp_net: Optional[pp.pandapowerNet] = None
        self._pp_lock = threading.RLock()
        self._last_update = 0
        self._last_pre_sim = 0
        self._simulation_iteration = 0
        self._processors: List[Processor] = []
        self._enable_measurement_read_from_pandapower = False

    @property
    def simulation_iteration(self):
        return self._simulation_iteration

    def _get_time_since_last_pre_sim_processing(self, passed_time: Optional[float]):
        if passed_time is not None:
            return passed_time
        # TODO: Use Wattson Time?
        if self._last_pre_sim == 0:
            return None
        return time.time() - self._last_pre_sim

    def _get_step_duration(self, passed_time: Optional[float] = None):
        if passed_time is not None:
            return passed_time
        # TODO: Use Wattson Time?
        if self._last_update == 0:
            return None
        return time.time() - self._last_update

    def simulate(self, passed_time: Optional[float] = None) -> bool:
        if self._pp_net is None:
            self.prepare_simulator()
            self._enable_measurement_read_from_pandapower = True

        self._apply_pre_sim_processing(self._get_time_since_last_pre_sim_processing(passed_time))
        self._last_pre_sim = time.time()
        with self._pp_lock:
            sim_net = copy.deepcopy(self._pp_net)
            # Apply Pre-Sim noise
            if self._post_sim_noise_callback is not None:
                for table in pp.toolbox.pp_elements(other_elements=True, res_elements=False):
                    if not table.startswith("res_") and isinstance(sim_net[table], pd.DataFrame):
                        sim_net[table] = sim_net[table].apply(self._pp_pre_sim_noise, axis=1, result_type="expand", table=table)

        actions = [ProcessorReturnAction.KEEP_SIMULATING]
        try:
            while ProcessorReturnAction.KEEP_SIMULATING in actions:
                pp.runpp(sim_net)
                with self._pp_lock:
                    for table in pp.toolbox.pp_elements(other_elements=False, res_elements=True):
                        if table.startswith("res_") and isinstance(sim_net[table], pd.DataFrame):
                            self._pp_net[table] = sim_net[table]
                            # Apply Post-Sim noise
                            if self._post_sim_noise_callback is not None:
                                self._pp_net[table] = self._pp_net[table].apply(self._pp_post_sim_noise, axis=1, result_type="expand", table=table)
                    self._last_update = time.time()
                actions = self._apply_post_sim_processing(self._get_step_duration(passed_time))
                # TODO: Remove break?
                break
            self._simulation_iteration += 1
        except Exception as e:
            warnings.warn(f"{e}")
            warnings.warn(f"{traceback.format_exc()}")
            return False
        return True

    def _apply_pre_sim_processing(self, step_duration: Optional[float] = None):
        for processor in self._processors:
            try:
                processor.apply_pre_simulation_processing(self, step_duration)
            except Exception as e:
                warnings.warn(f"Pre-Simulation-Processing {processor.__class__.__name__} failed: {e=}")

    def _apply_post_sim_processing(self, step_duration: Optional[float] = None):
        actions = []
        for processor in self._processors:
            try:
                actions.append(processor.apply_post_simulation_processing(self, step_duration))
            except Exception as e:
                warnings.warn(f"Post-Simulation-Processing {processor.__class__.__name__} failed: {e=}")
        return actions

    def _pp_pre_sim_noise(self, row, table):
        return self._pp_noise(row, table, self._pre_sim_noise_callback)

    def _pp_post_sim_noise(self, row, table):
        return self._pp_noise(row, table, self._post_sim_noise_callback)

    def _pp_noise(self, row, table, callback):
        index = row.name
        try:
            for column in row.keys():
                value = row[column]
                grid_value = self.get_grid_value_by_pandapower_path(table, index, column)
                if grid_value is not None:
                    unit, scale = self.extract_unit_and_scale(column)
                    if scale is not None and scale != Scale.NONE:
                        scaled_value = scale.to_scale(value, grid_value.scale)
                        scaled_value = callback(self.simulation_iteration, grid_value, scaled_value)
                        value = scale.from_scale(scaled_value, grid_value.scale)
                row[column] = value
            return row
        except Exception as e:
            import traceback
            warnings.warn(f"{traceback.format_exc()}")
            return row

    def is_prepared(self) -> bool:
        return self._pp_net is not None

    def prepare_simulator(self):
        if self.get_option("enable_storage_processing", True):
            self._processors.append(StorageProcessor())
        if self.get_option("enable_protection_emulation", False):
            self._processors.append(ProtectionProcessor(
                grid_model=self,
                trigger_delay_seconds=self.get_option("protection_trigger_delay_seconds"),
                trigger_threshold_factor=self.get_option("protection_trigger_threshold_factor")
            ))

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
                        if grid_value.simulator_context is None or grid_value.simulator_context is False:
                            continue
                        table, index, column = grid_value.simulator_context
                        _, scale = self.extract_unit_and_scale(column)
                        try:
                            value = grid_value.raw_get_value()
                            if isinstance(value, (GridElement, enum.Enum)):
                                continue
                            if scale != Scale.NONE and grid_value.scale != Scale.NONE:
                                value = grid_value.scale.to_scale(value, scale)
                            self._set_pp_value(grid_value, table, index, column, value)
                        finally:
                            continue
        # Set "connected" values for assets
        for e_type, elements in self.elements.items():
            element: GridElement
            for element in elements.values():
                try:
                    grid_value = element.get_config("connected")
                    value = grid_value.raw_get_value()
                    grid_value.set_value(value)
                except KeyError:
                    pass

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

    def _set_pp_value(self, grid_value: GridValue, table, index, column, value):
        with self._pp_lock:
            self._pp_net[table].at[index, column] = value

    def _get_pp_value(self, table, index, column) -> Any:
        with self._pp_lock:
            return self._pp_net[table].at[index, column]

    def _on_before_read_value(self, grid_value: GridValue):
        #if value.last_updated > self._last_update:
        #    return
        if grid_value.source is not None:
            try:
                source_value = grid_value.get_grid_element().get(grid_value.source[1], grid_value.source[0])
                value = source_value.get_value()
                if self._measurement_noise_callback is not None:
                    value = self._measurement_noise_callback(self.simulation_iteration, grid_value, value)
                grid_value.set_value(value)
            except KeyError:
                warnings.warn(f"Source value {grid_value.source} not found")
                return

        # Special values
        try:
            if self._handle_getting_special_value(grid_value):
                return
        except Exception as e:
            warnings.warn(f"_handle_getting_special_value failed: {e=}")
            return

        if self._enable_measurement_read_from_pandapower and grid_value.value_context in [GridValueContext.MEASUREMENT]:
            if grid_value.simulator_context is None:
                warnings.warn(f"No pandapower context given for {grid_value.name}")
                return
            if grid_value.simulator_context is False:
                return
            try:
                table, index, column = grid_value.simulator_context
            except ValueError as e:
                import traceback
                warnings.warn(traceback.format_exc())
                warnings.warn(grid_value.simulator_context)
                return

            unit, scale = self.extract_unit_and_scale(column)

            try:
                value = self._get_pp_value(table, index, column)
                scaled_value = value
                if scale is not None and scale != Scale.NONE:
                    scaled_value = grid_value.scale.from_scale(value, scale)
                if self._measurement_noise_callback is not None:
                    scaled_value = self._measurement_noise_callback(self.simulation_iteration, grid_value, scaled_value)
                grid_value.set_value(scaled_value)
            except KeyError:
                return

    def _on_set_value(self, grid_value: GridValue, old_value: Any, new_value: Any):
        if grid_value.value_context not in [GridValueContext.CONFIGURATION]:
            return
        """
        for target in grid_value.targets:
            try:
                target_grid_value = grid_value.get_grid_element().get(target[1], target[0])
                target_value = new_value
                if len(target) == 3:
                    # If tuple has a third argument, this is the rewrite function
                    target_value = target[2](new_value)
                target_grid_value.set_value(target_value)
            except KeyError:
                warnings.warn(f"Target value {target} not found")
        """
        # Handle special cases
        try:
            stop_handling, new_value = self._handle_setting_special_values(grid_value, old_value, new_value)
        except Exception as e:
            warnings.warn(f"_handle_setting_special_values failed: {e=}")
            return
        if stop_handling:
            return
        if grid_value.simulator_context is None:
            warnings.warn(f"No pandapower context given for {grid_value.name}")
            return
        if grid_value.simulator_context is False:
            return
        table, index, column = grid_value.simulator_context
        unit, scale = self.extract_unit_and_scale(column)

        if grid_value.value_type in [float, int]:
            new_value = grid_value.scale.to_scale(new_value, scale)
        else:
            new_value = grid_value.value
        self._set_pp_value(grid_value, table, index, column, new_value)

    def _handle_getting_special_value(self, grid_value: GridValue) -> bool:
        """
        Handles updating special values for reading.
        Returns whether the value has been updated (and no further processing is necessary).
        """
        grid_element = grid_value.get_grid_element()

        # GENERATOR TARGET POWER
        if isinstance(grid_element, (StaticGenerator, Generator, Storage)):
            if grid_value.name == "cos_phi":
                # Calculate cos_phi
                active_power = grid_element.get_measurement_value("active_power")
                reactive_power = grid_element.get_measurement_value("reactive_power")
                try:
                    cos_phi = active_power / math.sqrt(active_power ** 2 + reactive_power ** 2)
                except ZeroDivisionError:
                    return True
                grid_value.raw_set_value(cos_phi)
                return True
        return False

    def _calculate_target_power_with_profile(self, grid_element: GridAsset, dimension: str):
        target_power_value = grid_element.get_config_value(f"target_{dimension}_power", 0)
        maximum_power = grid_element.get_maximum_power(dimension)
        # Check if profiles are active
        if grid_element.get_config_value("profile_enabled", False):
            profile_percentage = grid_element.get_config_value(f"{dimension}_power_profile_percentage", None)
            if maximum_power is None:
                warnings.warn(f"Cannot apply {profile_percentage} as maximum {dimension} power is None")
                return target_power_value
            if profile_percentage is None:
                # No profile active
                return target_power_value
            target_power_value = maximum_power * (profile_percentage / 100)
        return target_power_value

    def _calculate_target_active_power_with_profile(self, grid_element: GridAsset):
        return self._calculate_target_power_with_profile(grid_element, "active")

    def _calculate_target_reactive_power_with_profile(self, grid_element: GridAsset):
        return self._calculate_target_power_with_profile(grid_element, "reactive")

    def _calculate_storage_target_active_power_limits(self, grid_element: Storage):
        current_charge_wh = grid_element.get_measurement_value("current_charge")
        maximum_charge_wh = grid_element.get_property_value("maximum_charge")
        minimum_charge_wh = grid_element.get_property_value("minimum_charge")

        minimum_target_value = None
        maximum_target_value = None

        if minimum_charge_wh is None or np.isnan(minimum_charge_wh):
            minimum_charge_wh = 0
        if current_charge_wh <= minimum_charge_wh:
            # Battery is empty - only loading allowed
            minimum_target_value = 0
        if maximum_charge_wh is None or np.isnan(maximum_charge_wh):
            pass
        elif current_charge_wh >= maximum_charge_wh:
            # Battery is full - only discharging allowed
            maximum_target_value = 0

        return minimum_target_value, maximum_target_value

    def _handle_setting_special_values(self, grid_value, old_value, new_value) -> Tuple[bool, Any]:
        """
        Handle those values that (might) affect other values.
        Returns a tuple:
            bool indicating whether the handling should stop
            new_value replacement (i.e., which value should be passed to the simulator)
        """
        grid_element = grid_value.get_grid_element()

        # GRID ASSET CONNECTION
        if isinstance(grid_element, GridAsset):
            if grid_value.name == "connected":
                # When disconnecting a grid asset, we force its scaling to 0
                try:
                    if new_value:
                        grid_element.get_config("scaling").unfreeze()
                    else:
                        grid_element.get_config("scaling").freeze(0)
                finally:
                    self.notify_simulation_configuration_changed()
                    return True, new_value

            if grid_value.name == "profile_enabled":
                target_active_power = grid_element.get_config("target_active_power")
                target_reactive_power = grid_element.get_config("target_reactive_power")

                target_reactive_power.set_value(target_reactive_power.get_value())
                target_active_power.set_value(target_active_power.get_value())

        # Storage state of charge
        if isinstance(grid_element, Storage):
            if grid_value.name == "current_charge":
                # Update target active power and target reactive power to respect current charge
                target_active_power = grid_element.get_config("target_active_power")
                target_reactive_power = grid_element.get_config("target_reactive_power")
                target_reactive_power.set_value(target_reactive_power.get_value())
                target_active_power.set_value(target_active_power.get_value())

        # GENERATOR TARGET POWER
        if isinstance(grid_element, (StaticGenerator, Storage, Load)):
            if grid_value.name == "target_active_power_percentage":
                # Update target_active_power based on percentage
                max_power = grid_element.get_maximum_active_power()
                if max_power is None:
                    warnings.warn(f"No maximum_active_power set for {grid_element.get_identifier()} - cannot set percentage")
                    return True, None

                fraction = new_value / 100
                if np.isnan(fraction) or fraction < -1 or fraction > 1:
                    warnings.warn(f"Invalid percentage target active power for {grid_element.get_identifier()}: {new_value} ({fraction}) - falling back to 0")
                    fraction = 0
                target_active_power_value = max_power * fraction
                # Update target_active_power
                grid_element.get_config("target_active_power").set_value(target_active_power_value)
                self.notify_simulation_configuration_changed()
                return True, new_value

            if grid_value.name == "target_active_power":
                # Handle active power limits
                maximum_active_power = grid_element.get_maximum_active_power()
                minimum_active_power = grid_element.get_property_value("minimum_active_power", None)
                active_power_limit = grid_element.get_config_value("active_power_limit", None)
                target_active_power_value = self._calculate_target_active_power_with_profile(grid_element)

                # Limit by maximum_active_power (device constraints)
                if maximum_active_power is not None and not np.isnan(maximum_active_power):
                    target_active_power_value = min(target_active_power_value, maximum_active_power)
                # Limit with active_power_limit (e.g., wind or solar power)
                if active_power_limit is not None and not np.isnan(active_power_limit):
                    target_active_power_value = min(target_active_power_value, active_power_limit)
                # Respect minimum (if any)
                if minimum_active_power is not None and not np.isnan(minimum_active_power):
                    target_active_power_value = max(target_active_power_value, minimum_active_power)

                # Check for state of charge for storages
                if isinstance(grid_element, Storage):
                    storage_minimum_target, storage_maximum_target = self._calculate_storage_target_active_power_limits(grid_element)
                    if storage_minimum_target is not None:
                        target_active_power_value = max(target_active_power_value, storage_minimum_target)
                    if storage_maximum_target is not None:
                        target_active_power_value = min(target_active_power_value, storage_maximum_target)

                # Handle COS_PHI based on corrected value
                target_cos_phi_value = grid_element.get_config_value("target_cos_phi", None)
                reactive_power_mode = grid_element.get_config_value("reactive_power_mode", ReactivePowerMode.NONE)
                if target_cos_phi_value is not None:
                    if reactive_power_mode in [ReactivePowerMode.COS_PHI, ReactivePowerMode.BOTH]:
                        if 0 < target_cos_phi_value <= 1:
                            # Set reactive power based on cos phi
                            apparent_power = target_active_power_value / target_cos_phi_value
                            target_reactive_power_value = math.sqrt(apparent_power ** 2 - target_active_power_value ** 2)
                            grid_element.get_config("target_reactive_power").set_value(target_reactive_power_value)
                        else:
                            warnings.warn(f"Invalid target_cos_phi value given: {target_cos_phi_value} not in (0, 1]")
                self.notify_simulation_configuration_changed()
                return False, target_active_power_value

            if grid_value.name == "target_reactive_power":
                # Apply limits
                maximum_reactive_power = grid_element.get_maximum_reactive_power()
                reactive_power_limit = grid_element.get_config_value("reactive_power_limit", None)
                minimum_reactive_power = grid_element.get_property_value("minimum_reactive_power", None)
                target_reactive_power_value = self._calculate_target_reactive_power_with_profile(grid_element)

                # Limit by maximum_reactive_power (device constraints)
                if maximum_reactive_power is not None and not np.isnan(maximum_reactive_power):
                    target_reactive_power_value = min(target_reactive_power_value, maximum_reactive_power)
                # Limit with reactive_power_limit (e.g., wind or solar power)
                if reactive_power_limit is not None and not np.isnan(reactive_power_limit):
                    target_reactive_power_value = min(target_reactive_power_value, reactive_power_limit)
                # Respect minimum (if any)
                if minimum_reactive_power is not None and not np.isnan(minimum_reactive_power):
                    target_reactive_power_value = max(target_reactive_power_value, minimum_reactive_power)

                # Check for state of charge for storages
                if isinstance(grid_element, Storage):
                    storage_minimum_target, storage_maximum_target = self._calculate_storage_target_active_power_limits(grid_element)
                    if storage_minimum_target is not None:
                        target_reactive_power_value = max(target_reactive_power_value, storage_minimum_target)
                    if storage_maximum_target is not None:
                        target_reactive_power_value = min(target_reactive_power_value, storage_maximum_target)

                self.notify_simulation_configuration_changed()
                return False, target_reactive_power_value

            if grid_value.name == "target_cos_phi":
                target_cos_phi_value = new_value
                if new_value <= 0 or new_value > 1:
                    target_cos_phi_value = max(min(new_value, 1), 0.01)
                    warnings.warn(f"Invalid target_cos_phi requested ({new_value}) - using {target_cos_phi_value}")
                reactive_power_mode = grid_element.get_config_value("reactive_power_mode")
                if reactive_power_mode in [ReactivePowerMode.COS_PHI, ReactivePowerMode.BOTH]:
                    # Apply cos_phi by re-setting target_active_power
                    target_active_power = grid_element.get_config("target_active_power")
                    target_active_power.set_value(target_active_power.get_value())
                    self.notify_simulation_configuration_changed()
                return True, target_cos_phi_value

            """
            LIMITS OF RENEWABLES
            """
            if grid_value.name == "active_power_limit":
                # Acknowledge and re-set target_active_power
                target_active_power = grid_element.get_config("target_active_power")
                target_active_power.set_value(target_active_power.get_value())
                self.notify_simulation_configuration_changed()

            if grid_value.name == "reactive_power_limit":
                # Acknowledge and re-set target_reactive_power
                target_reactive_power = grid_element.get_config("target_reactive_power")
                target_reactive_power.set_value(target_reactive_power.get_value())
                self.notify_simulation_configuration_changed()

            """
            PROFILES
            """
            if grid_value.name == "active_power_profile_percentage":
                # Acknowledge and re-set target_active_power if profiles are enabled
                target_active_power = grid_element.get_config("target_active_power")
                target_active_power.set_value(target_active_power.get_value())
                self.notify_simulation_configuration_changed()

            if grid_value.name == "reactive_power_profile_percentage":
                # Acknowledge and re-set target_active_power if profiles are enabled
                target_reactive_power = grid_element.get_config("target_reactive_power")
                target_reactive_power.set_value(target_reactive_power.get_value())
                self.notify_simulation_configuration_changed()

        return False, new_value

    def _load_profiles(self, net):
        """
        Load profiles from pandapower network and transform them to date and time based structure
        """
        if "profiles" not in net:
            return

        def _transform_profile(df: pandas.DataFrame):
            profile = {}
            columns = df.columns
            for i, row in df.iterrows():
                datetime_string = row["time"]
                datetime_date = datetime.datetime.strptime(datetime_string, "%d.%m.%Y %H:%M")
                date_string = datetime_date.strftime("%m-%d")
                time_string = datetime_date.strftime("%H:%M:%S")
                values = {col: row[col] for col in columns if col != "time"}
                profile.setdefault(date_string, {})[time_string] = values

            return profile

        for key in net["profiles"]:
            self.profiles[key] = _transform_profile(net["profiles"][key])

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

        if "profiles" in net:
            self.profiles = {}
            self._load_profiles(net)

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
        if prefix == "switch" and "in_service" in attributes:
            del attributes["in_service"]
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
        e_type = table.replace("res_", "").replace("_est", "")
        if e_type == "ext_grid":
            return "external_grid"
        if e_type == "xward":
            return "extended_ward"
        return e_type

    def _add_simulator_context(self):
        for elements in self.elements.values():
            for element in elements.values():
                for attribute_name, attribute in element.get_grid_values():
                    if attribute.simulator_context is not None:
                        continue
                    column = PandaPowerGridModel.column_mapping().get(attribute_name)
                    if column is False or column is None:
                        attribute.simulator_context = column
                        continue
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
                    if col is False:
                        # Column has no pandapower equivalent
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
        if column is False:
            # Same as None, but desired mismatch
            return
        # Transcribe value
        if isinstance(v, GridElement):
            v = v.index
        elif isinstance(v, enum.Enum):
            v = PandaPowerGridModel._translate_enum_to_value(v)
        elif isinstance(v, Step):
            v = v.value
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
        if isinstance(value, StorageType):
            return value.value
        if isinstance(value, StaticGeneratorType):
            return value.value
        if isinstance(value, GeneratorType):
            return value.value
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
            return PandaPowerGridModel._parse_sgen_type(str(value))
        if element_type == "gen" and column == "type":
            return PandaPowerGridModel._parse_gen_type(str(value))
        if element_type in ["load", "sgen"] and column == "type":
            return {
                "wye": ConnectionType.WYE,
                "delta": ConnectionType.DELTA
            }.get(value, ConnectionType.NONE)
        if element_type == "storage" and column == "type":
            return PandaPowerGridModel._parse_storage_type(value)
        if element_type == "line" and column == "type":
            return {
                "cs": LineType.UNDERGROUND_CABLE,
                "ol": LineType.OVERHEAD_LINE
            }.get(value, LineType.NONE)

        raise ValueError(f"No Enum found for {element_type}.{column} ({value=})")

    @staticmethod
    def _parse_gen_type(value: str):
        value = value.lower()
        return {
            "sync": GeneratorType.SYNCHRONOUS,
            "async": GeneratorType.ASYNCHRONOUS,
        }.get(value, GeneratorType.NONE)

    @staticmethod
    def _parse_sgen_type(value: str):
        value = value.lower()
        extracted_type = {
            "pv": StaticGeneratorType.PHOTOVOLTAIC,
            "wp": StaticGeneratorType.WIND,

            "chp": StaticGeneratorType.COMBINED_HEATING_AND_POWER,
            "chp diesel": StaticGeneratorType.COMBINED_HEATING_AND_POWER_DIESEL,

            "residential fuel cell": StaticGeneratorType.FUEL_CELL_RESIDENTIAL,
            "fuel cell": StaticGeneratorType.FUEL_CELL,
            "fc": StaticGeneratorType.FUEL_CELL,
            "rfc": StaticGeneratorType.FUEL_CELL_RESIDENTIAL
        }.get(value, StaticGeneratorType.NONE)
        if extracted_type == StaticGeneratorType.NONE:
            # Manual select type
            if "biomass" in value:
                extracted_type = StaticGeneratorType.BIOMASS
                if "mv" in value:
                    extracted_type = StaticGeneratorType.BIOMASS_MV
                if "hv" in value:
                    extracted_type = StaticGeneratorType.BIOMASS_MV
            elif "wind" in value:
                extracted_type = StaticGeneratorType.WIND
                if "mv" in value:
                    extracted_type = StaticGeneratorType.WIND_MV
                if "hv" in value:
                    extracted_type = StaticGeneratorType.WIND_HV
            elif "hydro" in value:
                extracted_type = StaticGeneratorType.HYDRO
                if "mv" in value:
                    extracted_type = StaticGeneratorType.HYDRO_MV
                if "hv" in value:
                    extracted_type = StaticGeneratorType.HYDRO_HV
            elif "pv" in value:
                extracted_type = StaticGeneratorType.PHOTOVOLTAIC
                if "mv" in value:
                    extracted_type = StaticGeneratorType.PHOTOVOLTAIC_MV
                if "hv" in value:
                    extracted_type = StaticGeneratorType.PHOTOVOLTAIC_HV
            elif "res" in value:
                extracted_type = StaticGeneratorType.RESIDENTIAL
                if "pv" in value:
                    extracted_type = StaticGeneratorType.PHOTOVOLTAIC_RESIDENTIAL
        return extracted_type

    @staticmethod
    def _parse_storage_type(value):
        value = value.lower()
        extracted_type = StorageType.NONE
        if "pv" in value:
            extracted_type = StorageType.PV_STORAGE
            if "mv" in value:
                extracted_type = StorageType.PV_STORAGE_MV
            if "hv" in value:
                extracted_type = StorageType.PV_STORAGE_HV
            if "res" in value:
                extracted_type = StorageType.PV_STORAGE_RESIDENTIAL
        if "battery" in value:
            extracted_type = StorageType.BATTERY
        return extracted_type

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
            "std_type": "std_type",
            "generator_type": "type",
            "type": "type",
            "connection_type": "type",
            "element": "element",
            "zone": "zone",
            "position": "coords",
            "geo_position": "geodata",

            "nominal_power": "sn_mva",

            "active_power": "p_mw",
            "active_power_fix_point": "p_mw",
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
            "reactive_power_fix_point": "q_mvar",
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
            "rated_voltage": "vn_kv",

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
            "current_charge": False,
            "storage_type": "type",

            # Generator
            "active_power_limit": False,
            "reactive_power_limit": False,
            "target_cos_phi": False,
            "target_active_power_percentage": False,
            "cos_phi": False,
            "reactive_power_mode": False,
            "transformer": "power_station_trafo",

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
            "scaling": "scaling",
            "closed": "closed",
            "is_closed": "closed",

            "length": "length_km",

            "from_bus": "from_bus",
            "to_bus": "to_bus",
            "bus": "bus",
            "lv_bus": "lv_bus",
            "hv_bus": "hv_bus",

            # Profiles
            "profile_name": "profile",
            "active_power_profile_percentage": False,
            "reactive_power_profile_percentage": False,
            "profile_enabled": False,
            "explicit_control": False,
            "profile_percentage": False,

            # Protection
            "current_protection_enabled": False,

            # Shunt
            "step": "step",

            # Unset / unrelated
            "connected": False,
            "is_connected": False,
        }

    @staticmethod
    def reverse_column_mapping():
        cm = PandaPowerGridModel.column_mapping()
        rcm = {}
        for key, value in cm.items():
            rcm.setdefault(value, []).append(key)
        return rcm
