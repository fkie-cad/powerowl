import dataclasses

from .power_plant import PowerPlant


@dataclasses.dataclass(eq=False, kw_only=True)
class VirtualPowerPlant(PowerPlant):
    pass
