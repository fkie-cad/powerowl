import dataclasses
import hashlib
import warnings
from typing import List, TYPE_CHECKING, cast

from .configuration.mac_address import MacAddress
from .network_entity import NetworkEntity
from ...exceptions import DerivationError
from ...graph.constants import MAC_CACHE

if TYPE_CHECKING:
    from .interface import Interface


@dataclasses.dataclass(eq=False, kw_only=True)
class NetworkNode(NetworkEntity):
    def get_role(self) -> str:
        if self._attributes.get("role") is not None:
            return self._attributes["role"]
        return self.__class__.__name__.lower()
    
    def has_role(self, role: str):
        return self.get_role() == role or role in self._attributes.get("roles", [])

    def add_role(self, role: str):
        if self.has_role(role):
            return
        if self._attributes.get("role") is None:
            self._attributes["role"] = role
        else:
            self._attributes.setdefault("roles", []).append(role)

    def get_interfaces(self) -> List['Interface']:
        from .interface import Interface
        interfaces = self.mlg.get_intra_layer_neighbors(
            self,
            neighbor_instance_filter={Interface}
        )
        interfaces = [cast(Interface, interface) for interface in interfaces]
        for i, interface in enumerate(interfaces):
            interface.local_id = f"i{i}"
        return interfaces

    def set_local_interface_ids(self):
        self.get_interfaces()

    def get_base_mac(self, mac_prefix: str) -> MacAddress:
        mac_cache: dict = self.mlg.get(MAC_CACHE, {})
        mac_cache.setdefault(mac_prefix, {})
        prefixed_mac_cache: dict = mac_cache[mac_prefix]
        if self.uid in prefixed_mac_cache:
            return prefixed_mac_cache[self.uid]

        prefix_length = len(mac_prefix) / 2
        if prefix_length == 0:
            prefix_length = 0
        elif prefix_length == 1:
            prefix_length = 1
        else:
            raise ValueError("Invalid mac_prefix length")

        node_part_length = 4 - prefix_length
        i = 0
        mac = MacAddress()
        mac.set_bytes_from_string(mac_prefix)
        max_tries = 1000
        while True:
            hash_id = f"{self.uid}-{i}".encode("utf-8")
            node_hash = hashlib.sha256(hash_id)
            node_mac = node_hash.digest()[:node_part_length].hex()
            mac.set_bytes_from_string(node_mac, start_offset=prefix_length)
            if mac not in prefixed_mac_cache.values():
                prefixed_mac_cache[self.uid] = mac
                return mac
            warnings.warn("MAC collision occurred - determinism potentially affected")
            i += 1
            if i >= max_tries:
                raise DerivationError("Too many MAC collisions")
