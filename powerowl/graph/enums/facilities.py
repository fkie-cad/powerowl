import enum


class Facilities(str, enum.Enum):
    OFFICE = "office"
    CONTROL_CENTER = "control-center"

    SUBSTATION = "substation"
    TRANSFORMING_SUBSTATION = "transforming-substation"
    DISTRIBUTION_SUBSTATION = "distribution-substation"
    JUNCTION_BOX = "junction-box"
    POWER_PLANT = "power-plant"
    SOLAR_POWER_PLANT = "solar-power-plant"
    WIND_POWER_PLANT = "wind-power-plant"
    VIRTUAL_POWER_PLANT = "wind-power-plant"

    GENERIC = "generic"

    def short_name(self) -> str:
        return {
            Facilities.OFFICE: "office",
            Facilities.CONTROL_CENTER: "CC",

            Facilities.SUBSTATION: "SubStation",
            Facilities.TRANSFORMING_SUBSTATION: "TSS",
            Facilities.DISTRIBUTION_SUBSTATION: "DSS",
            Facilities.JUNCTION_BOX: "JB",
            Facilities.POWER_PLANT: "PowerP",
            Facilities.SOLAR_POWER_PLANT: "SolarP",
            Facilities.WIND_POWER_PLANT: "WindP",
            Facilities.VIRTUAL_POWER_PLANT: "VPP"
        }.get(self, self.value)
