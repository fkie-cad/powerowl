import abc
from typing import Optional

from powerowl.layers.network.configuration.providers.provider_name import ProviderName


class ProviderInfo(abc.ABC):
    def __init__(self, provider_name: ProviderName):
        self.provider_name = provider_name
        self._domain = "source"
        self._coupling = None
        self._transform = None

    @property
    def domain(self):
        return self._domain

    @property
    def export_domain(self):
        return f"{self._domain}s"

    @domain.setter
    def domain(self, domain: str):
        domain = domain.lower()
        if domain not in ["source", "target"]:
            raise ValueError("Invalid domain. Must be 'source' or 'target'")
        self._domain = domain

    @property
    def coupling(self) -> Optional[str]:
        return self._coupling

    @coupling.setter
    def coupling(self, coupling: Optional[str]):
        if coupling is not None:
            coupling = coupling.lower()
            if self.domain != "target":
                raise ValueError("'coupling' only valid for 'target' provider")
        self._coupling = coupling

    @property
    def transform(self) -> Optional[str]:
        return self._transform

    @transform.setter
    def transform(self, transform: Optional[str]):
        if transform is not None:
            transform = transform.lower()
            if self.domain != "source":
                raise ValueError("'transform' only valid for 'source' provider")
        self._transform = transform

    def to_dict(self) -> dict:
        return {
            "name": self.provider_name,
            "domain": self.domain,
            "coupling": self.coupling,
            "transform": self.transform
        }

    def from_dict(self, d: dict):
        self.provider_name = d["name"]
        self.domain = d["domain"]
        self.coupling = d["coupling"]
        self.transform = d["transform"]

    @abc.abstractmethod
    def get_provider_data_dict(self, as_primitive: bool = False) -> dict:
        ...

    def get_provider_dict(self, as_primitive: bool = False) -> dict:
        provider_name = self.provider_name.name if as_primitive else self.provider_name
        d = {
            "domain": self.domain,
            "provider_type": provider_name,
            "provider_data": self.get_provider_data_dict(as_primitive),
            "coupling": None,
            "transform": None
        }
        for optional in ["coupling", "transform"]:
            if d[optional] is None:
                del d[optional]
        return d
