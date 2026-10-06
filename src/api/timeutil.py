"""
Timestamp normalisation for API responses.

The pipeline writes naive local timestamps (``pd.Timestamp.now().isoformat()``) into its JSON reports, while MLflow
stores epoch milliseconds. A browser cannot know which timezone a naive string refers to, so every naive ISO string
in an API response is converted to an offset-aware one using the server's local timezone.
"""
import re
from datetime import datetime
from typing import Any

_NAIVE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$")


def localize_timestamps(value: Any) -> Any:
    """Recursively adds the server's UTC offset to naive ISO-8601 strings inside dicts / lists."""
    if isinstance(value, dict):
        return {k: localize_timestamps(v) for k, v in value.items()}
    if isinstance(value, list):
        return [localize_timestamps(v) for v in value]
    if isinstance(value, str) and _NAIVE_ISO.match(value):
        try:
            return datetime.fromisoformat(value).astimezone().isoformat()
        except ValueError:
            return value
    return value
