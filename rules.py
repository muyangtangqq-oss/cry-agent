"""Condition helpers shared by the signal extractors.

Scenario YAML rules use a deliberately small language:
  - numeric comparisons: "> 2.5", "< 0.5", ">= 38"
  - time windows:        "16:00-20:00", "22:00-06:00"
  - booleans:            "detected", "not_detected"

Keeping the grammar tiny makes the scenarios readable by non-engineers and
avoids embedding arbitrary Python in configuration files.
"""
import operator
import re
import time as _time

_COMPARATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

_NUMERIC = re.compile(r"^(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$")
_WINDOW = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def time_in_window(window: str, now: _time.struct_time = None) -> bool:
    """Check a HH:MM-HH:MM window, including windows crossing midnight."""
    match = _WINDOW.match(window.strip())
    if not match:
        return False

    start = int(match.group(1)) * 60 + int(match.group(2))
    end = int(match.group(3)) * 60 + int(match.group(4))
    if end % 60 == 0:
        # "16:00-20:00" is inclusive of the whole 20:00 hour, matching the
        # hour-granularity checks the prototype started with.
        end += 59
    now = now or _time.localtime()
    current = now.tm_hour * 60 + now.tm_min

    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def evaluate_condition(condition, value, now: _time.struct_time = None) -> bool:
    """Evaluate one scenario rule against a sensor/context value."""
    if condition is None:
        return True
    condition = str(condition).strip()
    if not condition:
        return True

    if _WINDOW.match(condition):
        return time_in_window(condition, now)

    match = _NUMERIC.match(condition)
    if match:
        if value is None or isinstance(value, bool):
            return False
        try:
            return _COMPARATORS[match.group(1)](float(value), float(match.group(2)))
        except (TypeError, ValueError):
            return False

    normalized = condition.lower()
    if normalized in ("detected", "true", "yes", "on"):
        return bool(value)
    if normalized in ("not_detected", "false", "no", "off"):
        return not bool(value)
    if normalized in ("present", "any"):
        return value is not None
    return False
