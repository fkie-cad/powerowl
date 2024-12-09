import enum


class VoltageNiveau(enum.Enum):
    NONE = "none"
    LV = "low_voltage"
    MV = "medium_voltage"
    HV = "high_voltage"
    UHV = "ultra_high_voltage"

    def as_value(self):
        if self == VoltageNiveau.LV:
            return 400
        elif self == VoltageNiveau.MV:
            return 10000
        elif self == VoltageNiveau.HV:
            return 110000
        else:
            return 220000

    @staticmethod
    def by_value(value):
        if value < 1000:
            return VoltageNiveau.LV
        if value < 50000:
            return VoltageNiveau.MV
        if value <= 110000:
            return VoltageNiveau.HV
        return VoltageNiveau.UHV

    def __hash__(self):
        return self.value.__hash__()

    def __lt__(self, other):
        if isinstance(other, VoltageNiveau):
            return self.value < other.value
        return False

    def __gt__(self, other):
        if isinstance(other, VoltageNiveau):
            return self.value > other.value
        return False

    def __eq__(self, other):
        if isinstance(other, VoltageNiveau):
            return self.value == other.value
        return False

    def __ne__(self, other):
        return self.value != other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value
