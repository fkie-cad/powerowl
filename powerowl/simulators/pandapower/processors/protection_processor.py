import logging
import time
import warnings

import numpy as np
from powerowl.layers.powergrid import PowerGridModel

from powerowl.layers.powergrid.elements import Switch, Line, DcLine
from powerowl.simulators.pandapower.processors.processor import Processor
from typing import TYPE_CHECKING, Optional, List, Union, Dict

from powerowl.simulators.pandapower.processors.processor_return_action import ProcessorReturnAction

if TYPE_CHECKING:
    from powerowl.simulators.pandapower import PandaPowerGridModel


class ProtectionProcessor(Processor):
    def __init__(self,
                 grid_model: 'PandaPowerGridModel',
                 trigger_delay_seconds: float = 60,
                 trigger_threshold_factor: float = 2,
                 trigger_threshold_reference: str = "line",
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)

        self._trigger_delay_seconds = trigger_delay_seconds
        if self._trigger_delay_seconds is None:
            self._trigger_delay_seconds = 60
        self._trigger_threshold_factor = trigger_threshold_factor
        if self._trigger_threshold_factor is None:
            self._trigger_threshold_factor = 2

        self._trigger_threshold_reference = trigger_threshold_reference
        if self._trigger_threshold_reference not in ["line", "switch"]:
            self._trigger_threshold_reference = "line"

        self._threshold_exceeded_timestamps = {}
        self._triggered_and_not_cleared = set()

        self._switch_current_thresholds: Dict[str, float] = {}
        self._observed_switches: List[Switch] = []
        self._switch_to_line_map: Dict[str, Union[Line, DcLine]] = {}
        self._create_mappings(grid_model)

    def _create_mappings(self, grid_model: 'PandaPowerGridModel'):
        # Check all switches
        switch: Switch
        for switch in grid_model.get_elements_by_type("switch"):
            if not switch.get_property_value("current_protection_enabled"):
                continue
            switch_max_i_ka = switch.get_property_value("maximum_current")
            line = switch.get_associated()
            if not isinstance(line, (Line, DcLine)):
                self.log(f"No line associated with {switch.get_identifier()}", level=logging.WARNING)
                continue
            line_max_i_ka = line.get_property_value("maximum_current")

            if self._trigger_threshold_reference == "line":
                if line_max_i_ka is None or np.isnan(line_max_i_ka):
                    self.log(f"No maximum current available for {switch.get_identifier()} ({line.get_identifier()})", level=logging.WARNING)
                    continue
                max_i_ka = line_max_i_ka * self._trigger_threshold_factor
            else:
                if switch_max_i_ka is None or np.isnan(switch_max_i_ka):
                    self.log(f"No maximum current available for {switch.get_identifier()}", level=logging.WARNING)
                    continue
                max_i_ka = switch_max_i_ka * self._trigger_threshold_factor

            self._observed_switches.append(switch)
            self._switch_to_line_map[switch.get_identifier()] = line
            self._switch_current_thresholds[switch.get_identifier()] = max_i_ka
            self.log(f"Observing {switch.get_identifier()} with {max_i_ka} for {self._trigger_delay_seconds} s")

    def _clear_trigger(self, grid_model: PowerGridModel, switch, now):
        # Clear potentially queued triggers
        if switch.get_identifier() in self._threshold_exceeded_timestamps:
            self.log(f"Switch {switch.get_identifier()} deceeds threshold at {now}")
        self._threshold_exceeded_timestamps.pop(switch.get_identifier(), None)
        if switch.get_identifier() in self._triggered_and_not_cleared:
            self.log(f"Switch {switch.get_identifier()} protection resolved - clearing")
            self._triggered_and_not_cleared.remove(switch.get_identifier())
            grid_model.trigger_on_protection_equipment_cleared(switch, "current_over_load")

    def _check_triggers(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None):
        now = time.time()
        trigger_candidates = []

        for switch in self._observed_switches:
            if not switch.get_config_value("closed"):
                # Already opened, nothing to do
                continue
            line = self._switch_to_line_map[switch.get_identifier()]
            line_current = line.get_measurement_value("current")
            if line_current is None or np.isnan(line_current):
                # Clear if applicable
                self._clear_trigger(grid_model, switch, now)
                continue
            threshold = self._switch_current_thresholds[switch.get_identifier()]
            if line_current >= threshold:
                # Protection should trigger
                if switch.get_identifier() not in self._threshold_exceeded_timestamps:
                    self.log(f"Switch {switch.get_identifier()} exceeds threshold at {now}")
                self._threshold_exceeded_timestamps.setdefault(switch.get_identifier(), now)
                if now - self._threshold_exceeded_timestamps[switch.get_identifier()] >= self._trigger_delay_seconds:
                    trigger_candidates.append({
                        "switch": switch,
                        "line": line,
                        "threshold": threshold,
                        "current": line_current,
                        "overload": line_current / threshold
                    })
            else:
                # Clear potentially queued triggers
                self._clear_trigger(grid_model, switch, now)

        if len(trigger_candidates) > 0:
            trigger_candidates = sorted(trigger_candidates, key=lambda c: c["overload"], reverse=True)
            trigger_candidate = trigger_candidates[0]

            switch = trigger_candidate["switch"]
            self.log(f"Triggering {switch.get_identifier()}: {trigger_candidate['line'].get_identifier()} is at {trigger_candidate['current']} / {trigger_candidate['threshold']} @ {now}")
            switch.get_config("closed").set_value(False)
            self._threshold_exceeded_timestamps.pop(switch.get_identifier(), None)
            self._triggered_and_not_cleared.add(switch.get_identifier())
            grid_model.trigger_on_protection_equipment_triggered(switch, "current_over_load")
            return True

        return False

    def apply_pre_simulation_processing(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None):
        self._check_triggers(grid_model, simulation_step_interval)

    def apply_post_simulation_processing(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None):
        if self._check_triggers(grid_model, simulation_step_interval):
            return ProcessorReturnAction.KEEP_SIMULATING
