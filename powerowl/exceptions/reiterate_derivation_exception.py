from powerowl.layers.powergrid import PowerGridModel
from typing import Optional


class ReiterateDerivationException(Exception):
    def __init__(self, power_grid_model: PowerGridModel, updated_configuration: dict, updated_kwargs: Optional[dict] = None):
        super().__init__("ReiterateDerivationException")
        self.power_grid_model: PowerGridModel = power_grid_model
        self.updated_configuration: dict = updated_configuration
        if updated_kwargs is None:
            updated_kwargs = {}
        self.updated_kwargs: dict = updated_kwargs

