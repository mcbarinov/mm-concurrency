"""Asynchronous utility functions for concurrent programming."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


@asynccontextmanager
async def async_timeout(seconds: float) -> AsyncGenerator[None]:
    """Context manager for timeout handling.

    Provides a clean, readable way to add timeouts to async operations.
    Similar to anyio.fail_after() but using standard library asyncio.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        TimeoutError: If the operation takes longer than specified timeout
        ValueError: If seconds is not positive

    Example:
        async with async_timeout(5.0):
            result = await slow_operation()

        # Can be nested and combined with other context managers
        async with async_timeout(10.0):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(task1())
                tg.create_task(task2())
    """
    if seconds <= 0:
        raise ValueError("Timeout must be positive")

    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError:
        # Convert to standard TimeoutError for consistency
        raise TimeoutError from None
