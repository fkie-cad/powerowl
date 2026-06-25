import enum


class Endian(str, enum.Enum):
    # ABCD
    BIG_ENDIAN = "big-endian"
    # DCBA
    LITTLE_ENDIAN = "little-endian"
    # BADC
    BIG_SWAP_ENDIAN = "big-swap-endian"
    # DCBA
    LITTLE_SWAP_ENDIAN = "little-swap-endian"
