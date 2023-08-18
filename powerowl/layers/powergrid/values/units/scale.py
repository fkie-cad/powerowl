import enum
from decimal import Decimal


class Scale(Decimal, enum.Enum):
    """
    Represents the scale of a value, e.g., Kilo or Mega.
    """
    NONE = 0

    MICRO = Decimal("1e-6")
    MILLI = Decimal("1e-3")
    BASE = Decimal("1e0")
    KILO = Decimal("1e3")
    MEGA = Decimal("1e6")
    GIGA = Decimal("1e9")
    TERA = Decimal("1e12")

    def get_prefix(self) -> str:
        """
        Returns the prefix for this scale as String, e.g., k for Kilo.
        :return: The prefix.
        """
        return {
            Scale.MICRO: "µ",
            Scale.MILLI: "m",
            Scale.NONE: "",
            Scale.BASE: "",
            Scale.KILO: "k",
            Scale.MEGA: "M",
            Scale.GIGA: "G",
            Scale.TERA: "T"
        }.get(self)

    def to_base(self, value) -> float:
        """
        Converts a value with this instance's scale to the base scale.
        :param value: The value to convert
        :return: The value in base scale
        """
        s = self.value
        if s == 0:
            s = 1
        s = Decimal(s)
        d = Decimal(value)
        return float(d * s)

    def from_base(self, value):
        """
        Converts the value given in base scale to this instance's scale.
        :param value: The value to convert
        :return: The value scaled to this instance's scale.
        """
        s = self.value
        if s == 0:
            s = 1
        s = Decimal(s ** -1)
        d = Decimal(value)
        return float(d * s)

    def from_scale(self, value: float, value_scale: 'Scale') -> float:
        base_value = value_scale.to_base(value)
        return self.from_base(base_value)

    def to_scale(self, value: float, value_scale: 'Scale') -> float:
        base_value = self.to_base(value)
        scaled = value_scale.from_base(base_value)
        return scaled
