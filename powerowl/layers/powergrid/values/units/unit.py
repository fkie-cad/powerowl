import enum


class Unit(enum.Enum):
    NONE = "None"
    VOLT = "Volt"
    AMPERE = "Ampere"
    WATT = "Watt"
    WATT_HOUR = "WattHour"
    VAR = "Var"
    PER_UNIT = "PerUnit"
    DEGREE = "Degree"
    METER = "Meter"
    PERCENT = "Percent"
    OHM = "Ohm"
    VOLT_AMPERE = "VoltAmpere"
    FARAD = "Farad"

    def get_symbol(self):
        return {
            Unit.NONE: "",
            Unit.VOLT: "V",
            Unit.AMPERE: "A",
            Unit.WATT: "W",
            Unit.VAR: "Var",
            Unit.PER_UNIT: "pu",
            Unit.DEGREE: "°",
            Unit.METER: "m",
            Unit.PERCENT: "%",
            Unit.OHM: "Ω",
            Unit.VOLT_AMPERE: "VA",
            Unit.FARAD: "F",
            Unit.WATT_HOUR: "Wh"
        }.get(self, "???")
