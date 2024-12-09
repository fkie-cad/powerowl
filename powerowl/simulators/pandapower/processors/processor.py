import abc
from typing import TYPE_CHECKING, Optional

from powerowl.simulators.pandapower.processors.processor_return_action import ProcessorReturnAction

if TYPE_CHECKING:
    from powerowl.simulators.pandapower import PandaPowerGridModel


class Processor(abc.ABC):
    def apply_pre_simulation_processing(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None):
        pass

    def apply_post_simulation_processing(self, grid_model: 'PandaPowerGridModel', simulation_step_interval: Optional[float] = None) -> ProcessorReturnAction:
        return ProcessorReturnAction.NONE
