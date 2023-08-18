import dataclasses

from .facility import Facility


@dataclasses.dataclass(eq=False, kw_only=True)
class Substation(Facility):
    pass
