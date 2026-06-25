import enum


class TlsVersion(str, enum.Enum):
    NONE = "NONE"
    SSL_3_0 = "SSL_3_0"
    TLS_1_0 = "TLSv1.0"
    TLS_1_1 = "TLSv1.1"
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"

    def get_number(self) -> int:
        if self == TlsVersion.NONE:
            return 0
        elif self == TlsVersion.SSL_3_0:
            return 3
        elif self == TlsVersion.TLS_1_0:
            return 4
        elif self == TlsVersion.TLS_1_1:
            return 5
        elif self == TlsVersion.TLS_1_2:
            return 6
        elif self == TlsVersion.TLS_1_3:
            return 7
        return 0

    @classmethod
    def parse_string(cls, string: str) -> "TlsVersion":
        s = string.strip().lower()
        if s == "none" or s == "":
            return TlsVersion.NONE
        if s in ["1.2", "tls_1_2", "tlsv1.2"]:
            return TlsVersion.TLS_1_2
        if s in ["1.3", "tls_1_3", "tlsv1.3"]:
            return TlsVersion.TLS_1_3
        return TlsVersion.NONE
