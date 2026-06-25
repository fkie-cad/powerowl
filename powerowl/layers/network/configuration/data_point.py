from dataclasses import dataclass, field
from typing import Optional, List, Type

from powerowl.graph.model_node import ModelNode
from powerowl.layers.network.configuration.data_point_type import DataPointType
from powerowl.layers.network.configuration.data_point_value import DataPointValue
from powerowl.layers.network.configuration.protocols.protocol_info import ProtocolInfo
from powerowl.layers.network.configuration.providers.power_grid_provider_info import PowerGridProviderInfo
from powerowl.layers.network.configuration.providers.provider_info import ProviderInfo
from powerowl.layers.powergrid.values.grid_value import GridValue
from powerowl.layers.powergrid.values.grid_value_type import Step
from powerowl.layers.powergrid.values.units.scale import Scale
from powerowl.layers.powergrid.values.units.unit import Unit


@dataclass(kw_only=True)
class DataPoint(ModelNode):
    _data_point_id: Optional[str] = None
    description: Optional[str] = None
    value_type: Type[DataPointValue] = None
    value: DataPointValue = None
    scale: Scale = Scale.NONE
    unit: Unit = Unit.NONE
    protocol: Optional[ProtocolInfo] = None
    providers: List[ProviderInfo] = field(default_factory=list)
    coupling: Optional[str] = None
    related_points: List['DataPoint'] = field(default_factory=list)

    grid_values: List[GridValue] = field(default_factory=list)
    server: ModelNode = None

    @property
    def data_point_id(self):
        if self._data_point_id is None and self.protocol is not None:
            return self.protocol.generate_data_point_id()
        return self._data_point_id

    def get_identifier(self):
        return self.data_point_id

    def to_data_point_dict(self, as_primitive: bool = False) -> dict:
        # Derive dictionary for source and target providers (domain)
        providers = {}
        provider: ProviderInfo
        for provider in self.providers:
            providers.setdefault(provider.export_domain, []).append(provider.get_provider_dict(as_primitive=as_primitive))
        value = self.value
        if isinstance(value, Step):
            value = value.value
        # Build information dict
        d = {
            "identifier": self.data_point_id,
            "type": self.get_data_point_type().value,
            "description": self.description,
            "value": value,
            "providers": providers,
            "scale": self.scale.name,
            "unit": self.unit.name,
            "coupling": self.coupling,
            "related": list(set([dp.data_point_id for dp in self.related_points if dp.data_point_id != self.data_point_id]))
        }
        # Add Protocol Information
        d.update(self.protocol.get_protocol_dict(as_primitive=as_primitive))
        if self.server is not None:
            d["protocol_server_id"] = self.server.uid
        # Add PowerGrid Information if applicable
        ## Direct grid values
        direct_grid_value_identifiers = []
        for provider in self.providers:
            if isinstance(provider, PowerGridProviderInfo):
                direct_grid_value_identifiers.append(f"{provider.element_id}.{provider.attribute_context}.{provider.attribute_name}")
        ## Indirect
        grid_value_via_map = {}
        for related in set(self.related_points):
            if related == self:
                continue
            for grid_value in related.grid_values:
                grid_value_via_map[grid_value.get_identifier()] = {
                    "device": related.name,
                    "protocol": related.protocol.name.value,
                }

        ## Associated and direct ones
        d["grid_value"] = False
        if len(self.grid_values) > 0:
            grid_value_descriptions = [{
                "grid_element": grid_value.get_grid_element().get_identifier(),
                "context": grid_value.value_context.name,
                "attribute": grid_value.name,
                "type": str(grid_value.value_type.__name__),
                "direct": grid_value.get_identifier() in direct_grid_value_identifiers,
                "via": grid_value_via_map.get(grid_value.get_identifier(), False)
            } for grid_value in self.grid_values]
            if len(grid_value_descriptions) == 1:
                d["grid_value"] = grid_value_descriptions[0]
            else:
                d["grid_value"] = grid_value_descriptions

        # Delete empty keys that are optional
        for optional in ["description", "coupling"]:
            if d[optional] is None:
                del d[optional]
        return d

    def is_monitoring(self):
        return not self.is_control()

    def is_control(self):
        return len([p for p in self.providers if p.domain == "target"])

    def get_description_lines(self) -> List[str]:
        description_lines = [
            f"Protocol: {self.protocol.name.value}",
        ]
        for key, value in self.protocol.get_protocol_data_dict().items():
            description_lines.append(f" {key}: {value}")
        for provider in self.providers:
            description_lines.append(f"Provider: {provider.provider_name.value}")
            for key, value in provider.get_provider_data_dict().items():
                description_lines.append(f" {key}: {value}")
        return description_lines

    def __hash__(self):
        return hash(self.data_point_id)

    def __eq__(self, other):
        if isinstance(other, DataPoint):
            return other.data_point_id == self.data_point_id
        return False

    def get_data_point_type(self) -> DataPointType:
        return DataPointType.DATA_POINT
