"""XML-based parsers for native Apple Health exports."""

from .activity_parser import ActivityParser
from .heart_rate_parser import HeartRateParser
from .sleep_parser import SleepParser
from .streaming_adapter import StreamingXMLParser

# Try to import fast parser
try:
    from .fast_streaming_parser import FastStreamingXMLParser
    __all__ = [
        "SleepParser",
        "ActivityParser",
        "HeartRateParser",
        "StreamingXMLParser",
        "FastStreamingXMLParser",
    ]
except ImportError:
    __all__ = [
        "SleepParser",
        "ActivityParser",
        "HeartRateParser",
        "StreamingXMLParser",
    ]
