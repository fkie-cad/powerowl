import warnings

import numpy as np

from powerowl.simulators.pandapower.processors.processor import Processor
from typing import TYPE_CHECKING, Optional

from powerowl.simulators.pandapower.processors.processor_return_action import ProcessorReturnAction

if TYPE_CHECKING:
    from powerowl.simulators.pandapower import PandaPowerGridModel


class StorageProcessor(Processor):
    def apply_pre_simulation_processing(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None):
        if simulation_step_interval is None or simulation_step_interval == 0:
            return

        # Calculate storage state-of-charge
        for storage in grid_model.get_elements_by_type("storage"):
            current_charge_config = storage.get_config("current_charge")
            current_charge_measurement = storage.get_measurement("current_charge")

            current_active_power = storage.get_measurement_value("active_power", 0)

            maximum_charge_wh = storage.get_property_value("maximum_charge")
            minimum_charge_wh = storage.get_property_value("minimum_charge")
            if minimum_charge_wh is None or np.isnan(minimum_charge_wh):
                minimum_charge_wh = 0
                storage.get_property("minimum_charge").set_value(0)

            if np.isnan(current_active_power) or current_active_power is None:
                continue
            if current_active_power == 0:
                # No changes when no active power is consumed / fed in
                continue
            current_charge_wh = current_charge_measurement.get_value()
            if current_charge_wh is None:
                current_charge_wh = 0
            time_passed_hours = simulation_step_interval / 3600
            charge_change_wh = current_active_power * time_passed_hours
            new_charge_wh = current_charge_wh + charge_change_wh
            # Potentially limit charge
            new_charge_wh = max(minimum_charge_wh, new_charge_wh)
            if maximum_charge_wh is None or np.isnan(maximum_charge_wh) or maximum_charge_wh == 0:
                warnings.warn(f"No capacity / maximum charge given for {storage.get_identifier()}")
            else:
                new_charge_wh = min(maximum_charge_wh, new_charge_wh)
                charge_percentage = (new_charge_wh / maximum_charge_wh) * 100
                storage.get_config("state_of_charge").set_value(charge_percentage)
            current_charge_config.set_value(new_charge_wh)

    def apply_post_simulation_processing(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None) -> ProcessorReturnAction:
        return ProcessorReturnAction.NONE
