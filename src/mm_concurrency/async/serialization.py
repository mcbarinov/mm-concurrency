"""Asynchronous serialization decorators for async functions.

Provides decorator:
- async_serialized: All calls to the async function are serialized
"""

import asyncio
import functools
from collections.abc import Awaitable, Callable


def async_serialized[T, **P](func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator that ensures all calls to an async function are executed serially.

    Creates a single asyncio.Lock for the function, guaranteeing that only one
    coroutine can execute the function at any time, regardless of arguments.
    Other coroutines will wait for their turn.

    Args:
        func: Async function to serialize

    Returns:
        Serialized version of the async function with the same signature

    Example:
        @async_serialized
        async def update_global_state() -> None:
            # Only one coroutine can execute this at a time
            global_counter += 1

        @async_serialized
        async def critical_section(data: dict) -> str:
            # All calls serialized, even with different arguments
            return await process_shared_resource(data)
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with lock:
            return await func(*args, **kwargs)

    return wrapper
