import math
import re
import textwrap
import threading
import time
from typing import Dict


class Timing:
    enabled: bool = False
    print_enabled: bool = True
    line_width: int = 50
    gap: int = 5
    format_as_seconds: bool = True
    sub_timing_visibility_level = 1

    _parents: Dict[int, 'Timing'] = {}

    def __init__(self, message: str):
        self.message = message
        self._has_sub_timings = False
        self._has_visible_sub_timing: bool = False
        self._tid = threading.get_ident()
        self._parent = Timing._parents.get(self._tid, None)
        self._level = 0 if self._parent is None else self._parent.get_level() + 1
        self._start_time = -1
        self._start_time_ns = -1
        self._end_time = -1
        self._end_time_ns = -1
        self._duration = -1
        self._duration_ns = -1
        self._description_printed = False
        self._sum_timings = {}
        self._as_sum_timing: bool = False

    def __enter__(self):
        if not Timing.enabled:
            return
        Timing._parents[self._tid] = self
        if self._parent is not None:
            self._parent.start_sub_timing(self)
        self._start_time = time.time()
        self._start_time_ns = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not Timing.enabled:
            return
        self._end_time_ns = time.perf_counter_ns()
        self._end_time = time.time()
        Timing._parents[self._tid] = self._parent
        if self._parent is not None:
            self._parent.end_sub_timing(self)
        self._duration_ns = self._end_time_ns - self._start_time_ns
        self._duration = self._duration_ns / 10**9

        if self._as_sum_timing:
            if self._parent is None and self.print_enabled:
                self.auto_print()
            else:
                self._parent.add_sum_timing(self.message, self._duration_ns)
        elif self.print_enabled:
            self.auto_print()

    def add_sum_timing(self, name: str, duration_ns: float):
        self._sum_timings.setdefault(name, {"count": 0, "sum_duration": 0})
        self._sum_timings[name]["sum_duration"] += duration_ns
        self._sum_timings[name]["count"] += 1

    def as_sum_timing(self) -> 'Timing':
        self._as_sum_timing = True
        return self

    def is_visible(self):
        return self._level <= Timing.sub_timing_visibility_level

    def start_sub_timing(self, sub_timing: 'Timing'):
        self._has_visible_sub_timing |= sub_timing.is_visible()
        if not self._description_printed and self._has_visible_sub_timing and Timing.print_enabled:
            self.print_description(end=True)
        self._has_sub_timings = True

    def end_sub_timing(self, sub_timing: 'Timing'):
        pass

    def reset(self):
        self._start_time = -1
        self._start_time_ns = -1
        self._end_time = -1
        self._end_time_ns = -1
        self._duration = -1
        self._duration_ns = -1

    def get_indent(self):
        if self._parent is None:
            return ""
        return self._parent.get_indent() + "| "

    def auto_print(self):
        if not self.is_visible():
            return
        if self._has_visible_sub_timing:
            print(f"{self.get_indent()}| ".ljust(Timing.line_width + Timing.gap, " "))
        self.print()
        for name, info in self._sum_timings.items():
            duration_ns = info["sum_duration"]
            count = info["count"]
            print(f"{self.get_indent()}   Sum: {name}".ljust(Timing.line_width + Timing.gap), end="")
            print(f"{self.format_timespan_ns(duration_ns)}   ({count} samples)")
        if self._level == 0:
            print(f"{self.get_indent()}".ljust(Timing.line_width + Timing.gap, "-"))

    def print(self):
        if not self.is_visible():
            return
        self.print_description()
        self.print_timing()

    def print_description(self, end=False):
        self._description_printed = True
        primary_indent = self.get_indent()
        subsequent_indent = primary_indent + " "
        lines = textwrap.wrap(self.message, width=Timing.line_width, initial_indent=primary_indent,
                              subsequent_indent=subsequent_indent)
        for i, line in enumerate(lines, 1):
            if i < len(lines):
                print(line)
            else:
                print(line.ljust(Timing.line_width + Timing.gap), end="")
                if end:
                    print("")

    def print_timing(self):
        print(self.format_timespan_ns(self._duration_ns))

    def get_level(self):
        return self._level

    def get_duration_ns(self):
        return self._duration_ns

    def get_duration(self):
        return self._duration

    def format_duration(self, detail: int = 6) -> str:
        if self._duration_ns < 0:
            return "Not yet run"
        return self.format_timespan_ns(self._duration_ns)

    def format_timespan(self, seconds: float, detail: int = 6) -> str:
        return self.format_timespan_ns(int(seconds * 10**9), detail=detail)

    def get_string_precision(self, time_format_string: str) -> int:
        """
        Given a time in string format, calculates its precision (i.e., it counts the significant digits)
        """
        stripped = time_format_string.replace(".", "")
        stripped = re.sub(r'[^0-9]', ' ', stripped)
        sections = [e for e in stripped.split(" ") if len(e) > 0]
        numbers = [int(e) if int(e) > 0 else 0 for e in sections]
        lengths = [len(str(e)) for e in numbers]
        return sum(lengths)
    
    def get_color(self, seconds: float) -> str:
        try:
            from colorama import Fore, Back, Style
            if seconds < 0.5:
                return Fore.LIGHTBLACK_EX
            if seconds < 1.5:
                return Style.RESET_ALL
            if seconds < 3:
                return Fore.YELLOW
            return Fore.RED
        except:
            return ""
        
    def reset_color(self) -> str:
        try:
            from colorama import Style
            return Style.RESET_ALL
        except:
            return ""

    def format_timespan_ns(self, nanoseconds: int, detail: int = 6) -> str:
        if Timing.format_as_seconds:            
            seconds = nanoseconds * 10**-9
            formatted_as_seconds = f"{self.get_color(seconds=seconds)}{seconds:9.4f} s{self.reset_color()}"
            return formatted_as_seconds

        segments = []
        digits = 1 + int(math.log10(nanoseconds))
        shift_digits = digits - detail
        shift = 10**shift_digits
        unshift = 10**-shift_digits

        nanoseconds_shifted = int(int(nanoseconds / shift) / unshift)

        r = nanoseconds_shifted
        ns = r % 1000
        segments.insert(0, (ns, "ns"))
        r = int(r / 1000)
        micro_s = r % 1000
        segments.insert(0, (micro_s, "μs"))
        r = int(r / 1000)
        milli_s = r % 1000
        segments.insert(0, (milli_s, "ms"))
        r = int(r / 1000)
        seconds = r
        segments.insert(0, (seconds, "s"))
        formatted = []

        # Remove unused/zero fields
        while segments[-1][0] == 0:
            segments = segments[:-1]

        if nanoseconds_shifted == 0:
            return f"{self.get_color(0)}0ns{self.reset_color()}"

        for value, unit in segments:
            formatted.append(self.format_segment(value, unit))
        return f"{self.get_color(seconds=seconds)}{' '.join(formatted)}{self.reset_color()}"    

    def format_segment(self, value, unit) -> str:
        if unit == "s":
            return self.format_seconds(value)
        return f"{value}{unit}"

    def format_seconds(self, seconds: int) -> str:
        """
        Formats seconds as minutes and seconds
        """
        s = seconds % 60
        s = str(s).zfill(2)
        m = int(seconds / 60)
        if m == 0:
            return f"{s}s"
        m = str(m).zfill(2)
        return f"{m}m {s}s"
