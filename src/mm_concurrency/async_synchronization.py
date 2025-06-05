"""Asynchronous synchronization decorators for async functions.

Provides decorators:
- async_synchronized: All calls to the async function are synchronized
- async_synchronized_by_arg_value: Calls are synchronized only for matching argument values
"""

import asyncio
import functools
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable

import anyio


def async_synchronized[T, **P](func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator that ensures all calls to an async function are executed in synchronized manner.

    Creates a single anyio.Lock for the function, guaranteeing that only one
    coroutine can execute the function at any time, regardless of arguments.
    Other coroutines will wait for their turn.

    Args:
        func: Async function to synchronize

    Returns:
        Synchronized version of the async function with the same signature

    Example:
        @async_synchronized
        async def update_global_state() -> None:
            # Only one coroutine can execute this at a time
            global_counter += 1

        @async_synchronized
        async def critical_section(data: dict) -> str:
            # All calls synchronized, even with different arguments
            return await process_shared_resource(data)
    """
    lock = anyio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with lock:
            return await func(*args, **kwargs)

    return wrapper


def async_synchronized_by_arg_value[T, **P](
    index: int = 0, key: str | None = None, nonblocking: bool = False
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T | None]]]:
    """Decorator that synchronizes async function calls based on argument values.

    Each unique value of the specified argument gets its own asyncio.Lock, allowing
    concurrent execution for different argument values while synchronizing
    calls with the same argument value.

    Args:
        index: Index of the argument to use as the lock key (default: 0)
        key: Name of the parameter to use as the lock key (overrides index)
        nonblocking: If True, returns None when the lock is already held (default: False)

    Returns:
        Decorated async function that returns T or None (if nonblocking=True and lock is held)

    Raises:
        ValueError: If key is specified but not found in function signature

    Example:
        @async_synchronized_by_arg_value(index=0)
        async def process_user(user_id: str) -> None:
            # Only one coroutine can process the same user_id at a time
            # But different user_ids can be processed concurrently
            pass

        @async_synchronized_by_arg_value(key='user_id')
        async def process_user(user_id: str, data: dict) -> None:
            # More readable - synchronizes by user_id parameter
            pass

        @async_synchronized_by_arg_value(nonblocking=True)
        async def try_update_cache(cache_key: str) -> bool:
            # Returns None if another coroutine is already updating this key
            pass
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T | None]]:
        # Get function signature for key validation
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Validate key if provided
        if key and key not in param_names:
            raise ValueError(f"Parameter '{key}' not found in function signature. Available parameters: {param_names}")

        # Determine the index to use for key extraction
        target_index = param_names.index(key) if key else index

        # Shared state for all calls to this decorated function
        locks: dict[object, asyncio.Lock] = {}  # Maps lock keys to their asyncio.Lock objects
        usage_count: dict[object, int] = defaultdict(int)  # Reference counter for safe cleanup
        registry_lock = asyncio.Lock()  # Protects locks and usage_count from race conditions

        def extract_key(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
            """Extract the locking key from function arguments."""
            if key and key in kwargs:
                return kwargs[key]
            if target_index >= len(args):
                raise IndexError(f"Argument index {target_index} out of range. Function called with {len(args)} arguments.")
            return args[target_index]

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            lock_key = extract_key(args, kwargs)

            # Step 1: Atomically get or create lock and increment usage count
            # We increment BEFORE acquiring the actual lock to prevent cleanup race conditions
            async with registry_lock:
                if lock_key not in locks:
                    locks[lock_key] = asyncio.Lock()
                lock = locks[lock_key]
                usage_count[lock_key] += 1  # This coroutine is now "using" this lock

            # Step 2: Try to acquire the actual lock (different approach for blocking/nonblocking)
            if nonblocking and lock.locked():
                # Lock is already held in nonblocking mode - cleanup and return None
                async with registry_lock:
                    usage_count[lock_key] -= 1
                    if usage_count[lock_key] == 0:
                        # No other coroutines are using this lock, safe to remove
                        locks.pop(lock_key, None)
                        usage_count.pop(lock_key, None)
                return None

            # Step 3: Acquire the lock (blocking or already checked as available)
            async with lock:
                # Step 4: Execute function and cleanup
                try:
                    return await func(*args, **kwargs)
                finally:
                    # Cleanup usage tracking
                    # This is in finally to ensure cleanup even if func() raises an exception
                    async with registry_lock:
                        usage_count[lock_key] -= 1
                        if usage_count[lock_key] == 0:
                            # Last coroutine using this lock - safe to remove from memory
                            # This prevents memory leaks for functions with many unique lock keys
                            locks.pop(lock_key, None)
                            usage_count.pop(lock_key, None)

        return wrapper

    return decorator
