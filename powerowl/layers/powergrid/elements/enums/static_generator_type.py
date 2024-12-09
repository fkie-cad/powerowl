import enum

from powerowl.layers.powergrid.elements.enums.voltage_niveau import VoltageNiveau


class StaticGeneratorType(enum.Enum):
    NONE = "none"
    WIND = "WP"
    WIND_MV = "WP_MV"
    WIND_HV = "WP_HV"
    PHOTOVOLTAIC = "PV"
    PHOTOVOLTAIC_MV = "PV_MV"
    PHOTOVOLTAIC_HV = "PV_HV"
    PHOTOVOLTAIC_RESIDENTIAL = "PV_RES"
    RESIDENTIAL = "RES"
    HYDRO = "HYDRO"
    HYDRO_MV = "HYDRO_MV"
    HYDRO_HV = "HYDRO_HV"
    BIOMASS = "BIO"
    BIOMASS_MV = "BIO_MV"
    BIOMASS_HV = "BIO_HV"
    COMBINED_HEATING_AND_POWER = "CHP"
    COMBINED_HEATING_AND_POWER_DIESEL = "CHPD"
    FUEL_CELL = "FC"
    FUEL_CELL_RESIDENTIAL = "RFC"

    def get_voltage_niveau(self) -> VoltageNiveau:
        if "_HV" in self.name:
            return VoltageNiveau.HV
        if "_MV" in self.name:
            return VoltageNiveau.MV
        if "RESIDENTIAL" in self.name:
            return VoltageNiveau.LV
        return VoltageNiveau.NONE

    def get_clear_type(self) -> 'StaticGeneratorType':
        clear_type_name = self.name.replace("_HV", "").replace("_MV", "").replace("_RESIDENTIAL", "")
        try:
            clear_type = StaticGeneratorType[clear_type_name]
            return clear_type
        except KeyError:
            pass
        return self
