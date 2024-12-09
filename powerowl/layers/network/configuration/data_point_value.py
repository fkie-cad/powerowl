from typing import Union, Optional

from powerowl.layers.powergrid.values.grid_value_type import Step

DataPointValue = Optional[Union[str, int, float, bool, Step]]
