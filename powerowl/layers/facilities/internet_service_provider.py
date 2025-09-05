import dataclasses

from powerowl.graph.enums.owner import Owner

from .facility import Facility


@dataclasses.dataclass(eq=False, kw_only=True)
class InternetServiceProvider(Facility):
    def __post_init__(self):
        super().__post_init__()
        self.owner = Owner.INTERNET_SERVICE_PROVIDER
