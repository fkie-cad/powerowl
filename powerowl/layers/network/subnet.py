import dataclasses
import ipaddress
import math
import warnings
from typing import Optional, Set, Iterator

import networkx as nx

from powerowl.exceptions import DerivationError
from powerowl.exceptions.subnet_exhausted_error import SubnetExhaustedError
from powerowl.exceptions.undefined_network_error import UndefinedNetworkError
from powerowl.graph.enums import EdgeType, Layers
from powerowl.graph.enums.layer_matching_strategy import LayerMatchingStrategy
from powerowl.graph.model_node import ModelNode
from powerowl.layers.network.interface import Interface
from powerowl.layers.network.link import Link
from powerowl.layers.network.switch import Switch
from powerowl.performance.timing import Timing


@dataclasses.dataclass(eq=False, kw_only=True)
class Subnet(ModelNode):
    def __post_init__(self):
        super().__post_init__()
        self._ip_network: Optional[ipaddress.IPv4Network] = None
        self._used_subnets: Set[ipaddress.IPv4Network] = set()
        self._ip_address_generator: Optional[Iterator] = None

    def add_interface(self, interface: Interface):
        self.mlg.build_edge(self, interface, EdgeType.MEMBERSHIP)

    def get_interfaces(self) -> list[Interface]:
        network_layer = self.mlg.get_layer(Layers.NETWORK)
        return [
            interface for interface in
            self.mlg.get_inter_layer_neighbors(
                self, other_layers={network_layer},
                other_layers_matching_strategy=LayerMatchingStrategy.SHARED_TOP_LEVEL_LAYER
            )
            if isinstance(interface, Interface)
        ]

    def set_ip_network(self, ip_network: ipaddress.IPv4Network):
        self._ip_network = ip_network
        self._ip_address_generator = self._ip_network.hosts()

    def get_ip_network(self) -> ipaddress.IPv4Network:
        if self._ip_network is None:
            raise UndefinedNetworkError("This subnet has not yet been assigned an IP subnet")
        return self._ip_network

    def get_sub_ip_network(self, min_num_addresses: int, min_prefix_length: int = 24) -> ipaddress.IPv4Network:
        if self._ip_network is None:
            raise UndefinedNetworkError("This subnet has not yet been assigned an IP subnet")
        # Derive how many addresses are needed for the requested subnet
        # Always add at least two addresses as spare address (and broadcast)
        required_bits = math.ceil(math.log2(min_num_addresses + 2))
        # Respect minimum requested prefix length
        required_bits = max(required_bits, 32 - min_prefix_length)
        main_prefix_length = self._ip_network.prefixlen
        if main_prefix_length + required_bits > 32:
            raise DerivationError(
                f"Cannot derive a new subnet with {required_bits} "
                f"bits from a subnet with prefix length {main_prefix_length}"
            )
        prefix_length = 32 - required_bits
        potential_subnets = self._ip_network.subnets(new_prefix=prefix_length)
        suitable_subnet = None
        for potential_subnet in potential_subnets:
            if any([potential_subnet.overlaps(existing_subnet)
                    for existing_subnet in self._used_subnets]):
                continue
            suitable_subnet = potential_subnet
            break
        if suitable_subnet is None:
            raise SubnetExhaustedError(
                f"Cannot cut an additional subnet of size {prefix_length} from {self._ip_network}"
            )
        self._used_subnets.add(suitable_subnet)
        return suitable_subnet

    def reset_ip_generator(self):
        self._ip_address_generator = self._ip_network.hosts()

    def get_next_ip_address(self) -> ipaddress.IPv4Address:
        if self._ip_address_generator is None:
            raise UndefinedNetworkError(f"Not IP Subnet defined")
        address = next(self._ip_address_generator, None)
        if address is None:
            SubnetExhaustedError("No further addresses available in this subnet")
        return address

    def create_minimum_spanning_tree(self, layer_graph: Optional[nx.Graph] = None):
        """
        Derives a minimum spanning tree of all switches in the given subnet, i.e., it removes links that are
        responsible for loops between switches.
        """
        if layer_graph is None:
            layer_graph = self.mlg.get_layer_graph(self.mlg.get_layer(Layers.NETWORK), edge_type_filter={EdgeType.NETWORK_LINK})

        subnet_node_ids: Set[str] = set()
        for interface in self.get_interfaces():
            link = interface.get_network_link()
            if link is None:
                continue
            other_interface = link.get_other_interface(interface)
            node_a = interface.get_network_node()
            node_b = other_interface.get_network_node()
            if not isinstance(node_a, Switch) or not isinstance(node_b, Switch):
                continue
            subnet_node_ids.add(interface.uid)
            subnet_node_ids.add(other_interface.uid)
            subnet_node_ids.add(link.uid)
            subnet_node_ids.add(node_a.uid)
            subnet_node_ids.add(node_b.uid)
        subnet_graph = layer_graph.subgraph(subnet_node_ids)
        subnet_edges = sorted(sorted(e) for e in subnet_graph.edges())

        tree_edges = sorted(sorted(e) for e in nx.minimum_spanning_tree(subnet_graph).edges())

        for e in subnet_edges:
            if e not in tree_edges:
                model_edge = self.mlg.get_edge(e[0], e[1])
                if model_edge is None:
                    raise DerivationError(f"Cannot find edge {e}")
                if model_edge.edge_type != EdgeType.NETWORK_LINK:
                    raise DerivationError(f"Unexpected edge type {model_edge.edge_type}")
                link: Optional[Link] = None
                if isinstance(model_edge.node_a, Interface):
                    link = model_edge.node_a.get_network_link()
                if isinstance(model_edge.node_b, Interface):
                    link = model_edge.node_b.get_network_link()

                if link is None:
                    raise DerivationError(f"Cannot find a neighbored link at removed edge {model_edge}")
                interfaces = link.get_interfaces()
                if len(interfaces) != 2:
                    raise DerivationError(f"Link {link.uid} does not have 2 interfaces")
                interface_a = interfaces[0]
                interface_b = interfaces[1]
                self.mlg.remove_node(interface_a)
                self.mlg.remove_node(link)
                self.mlg.remove_node(interface_b)
