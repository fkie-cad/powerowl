import enum

from powerowl.layers.powergrid.elements.enums.voltage_niveau import VoltageNiveau


class StorageType(enum.Enum):
    NONE = "none"
    BATTERY = "battery"
    PV_STORAGE = "PV"
    PV_STORAGE_MV = "PV_MV"
    PV_STORAGE_HV = "PV_HV"
    PV_STORAGE_RESIDENTIAL = "PV_RES"

    def get_voltage_niveau(self) -> VoltageNiveau:
        """
        Returns the voltage niveau (level) based on the storage type
        """
        if self == StorageType.PV_STORAGE_MV:
            return VoltageNiveau.MV
        if self == StorageType.PV_STORAGE_HV:
            return VoltageNiveau.HV
        if self == StorageType.PV_STORAGE_RESIDENTIAL:
            return VoltageNiveau.LV
        return VoltageNiveau.NONE

    def get_clear_type(self) -> 'StorageType':
        """
        Returns the clear type without voltage niveau details
        """
        if self in [StorageType.PV_STORAGE_RESIDENTIAL, StorageType.PV_STORAGE_MV, StorageType.PV_STORAGE_HV, StorageType.PV_STORAGE]:
            return StorageType.PV_STORAGE
        return self
