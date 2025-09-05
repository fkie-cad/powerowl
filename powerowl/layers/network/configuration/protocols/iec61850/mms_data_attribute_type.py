import enum

from powerowl.layers.powergrid.values.grid_value_type import GridValueType, Step


class MmsDataAttributeType(int, enum.Enum):
    UNKNOWN_TYPE = -1
    BOOLEAN = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    INT128 = 5
    INT8U = 6
    INT16U = 7
    INT24U = 8
    INT32U = 9
    FLOAT32 = 10  # Float
    FLOAT64 = 11  # Double/float
    ENUMERATED = 12
    OCTET_STRING_64 = 13
    OCTET_STRING_6 = 14
    OCTET_STRING_8 = 15
    VISIBLE_STRING_32 = 16
    VISIBLE_STRING_64 = 17
    VISIBLE_STRING_65 = 18
    VISIBLE_STRING_129 = 19
    VISIBLE_STRING_255 = 20
    UNICODE_STRING_255 = 21
    TIMESTAMP = 22
    QUALITY = 23
    CHECK = 24
    CODEDENUM = 25
    GENERIC_BITSTRING = 26
    CONSTRUCTED = 27
    ENTRY_TIME = 28
    PHYCOMADDR = 29
    CURRENCY = 30
    OPTFLDS = 31
    TRGOPS = 32

    @staticmethod
    def from_primitive(value_type: GridValueType):
        if value_type == float:
            return MmsDataAttributeType.FLOAT32
        if value_type == int:
            return MmsDataAttributeType.INT32
        if value_type == bool:
            return MmsDataAttributeType.BOOLEAN
        if value_type == Step:
            return MmsDataAttributeType.INT32
        if value_type == str:
            return MmsDataAttributeType.UNICODE_STRING_255
        return MmsDataAttributeType.UNKNOWN_TYPE
