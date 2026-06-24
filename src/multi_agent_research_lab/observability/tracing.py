"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:
    _langsmith_traceable = None


def traceable(*args: Any, **kwargs: Any) -> Any:
    """Return LangSmith's decorator when available, otherwise a no-op decorator."""

    if _langsmith_traceable is not None:
        return _langsmith_traceable(*args, **kwargs)

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func: Any) -> Any:
        return func

    return decorator


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    The default lab implementation keeps traces in-process so students can inspect the workflow
    before wiring a hosted tracing backend.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
