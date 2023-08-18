from typing import Optional, List


class MacAddress:
    def __init__(self):
        self._mac_bytes = [0] * 6

    def to_int(self):
        byte_max = 2**8
        return sum(byte * byte_max**(len(self._mac_bytes)-i) for i, byte in enumerate(self._mac_bytes, 1))

    def __str__(self):
        return ":".join(hex(byte).lstrip("0x").zfill(2) for byte in self._mac_bytes)

    def __hash__(self):
        return self.to_int()

    def __eq__(self, other):
        if isinstance(other, MacAddress):
            return other.to_int() == self.to_int()
        return False

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return self == other or self > other

    def __gt__(self, other):
        if isinstance(other, MacAddress):
            return other.to_int() > self.to_int()
        return False

    def __lt__(self, other):
        if isinstance(other, MacAddress):
            return other.to_int() < self.to_int()
        return False

    def get_next_mac_address(self) -> 'MacAddress':
        return MacAddress.from_int(self.to_int() + 1)

    def get_previous_mac_address(self) -> 'MacAddress':
        return MacAddress.from_int(self.to_int() - 1)

    @staticmethod
    def from_int(mac_int: int) -> 'MacAddress':
        if 0 <= mac_int < 2**(6*8):
            mac_string = hex(mac_int).lstrip("0x").rjust(12, "0")
            return MacAddress.from_string(mac_string)
        raise ValueError(f"Mac Value out of range ({mac_int})")

    @staticmethod
    def from_string(mac_string: str) -> 'MacAddress':
        mac = MacAddress()
        mac.set_bytes_from_string(mac_string)
        return mac

    def set_byte(self, byte_index: int, byte: int):
        """
        Sets a single byte of this MAC
        """
        if byte < 0 or byte > 256:
            raise ValueError(f"Cannot set byte to {byte} - invalid value")
        if 0 <= byte_index < len(self._mac_bytes):
            self._mac_bytes[byte_index] = byte
            return True
        raise IndexError(f"Invalid byte ID {byte_index}")

    def set_bytes_from_string(self, byte_string: str, start_offset: int = 0, length: Optional[int] = None):
        """
        Sets one or multiple bytes from a string representation of the bytes.
        :param byte_string: The (hex) byte string to apply.
        :param start_offset: The index of the first byte to fill in.
        :param length: How many bytes to fill. If None, this is automatically determined
        """
        byte_list = self._parse_byte_string(byte_string=byte_string)
        i = start_offset
        if length is None:
            length = len(byte_list)
        max_byte = start_offset + length
        while i < max_byte:
            self.set_byte(i, byte_list.pop(0))
            i += 1

    def _parse_byte_string(self, byte_string: str) -> List[int]:
        byte_string = byte_string.replace(":", "").replace("-", "")
        byte_list = []
        if len(byte_string) > 12:
            raise ValueError(f"Invalid byte representation: {byte_string}")
        if len(byte_string) == 0:
            return []
        # Verify format
        int(byte_string, 16)
        while len(byte_string) > 0:
            byte = byte_string[-2:]
            byte_string = byte_string[:-2]
            byte_list.insert(0, int(byte, 16))
        return byte_list
