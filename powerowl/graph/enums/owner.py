import enum


class Owner(str, enum.Enum):
    NONE = "NONE"
    GRID_OPERATOR = "Grid Operator"
    CUSTOMER = "Customer"
    INTERNET_SERVICE_PROVIDER = "Internet Service Provider"
