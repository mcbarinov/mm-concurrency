"""Thread synchronization decorators for synchronizing function calls.

Provides two decorators:
- synchronized: All calls to the function are synchronized
- synchronized_by_arg_value: Calls are synchronized only for matching argument values
"""

import functools
import inspect
from collections import defaultdict
from collections.abc import Callable
from threading import Lock, RLock


def synchronized_by_arg_value[T, **P](
    index: int = 0, key: str | None = None, nonblocking: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Decorator that synchronizes function calls based on argument values.

    Each unique value of the specified argument gets its own lock, allowing
    concurrent execution for different argument values while synchronizing
    calls with the same argument value.

    Args:
        index: Index of the argument to use as the lock key (default: 0)
        key: Name of the parameter to use as the lock key (overrides index)
        nonblocking: If True, returns None when the lock is already held (default: False)

    Returns:
        Decorated function that returns T or None (if nonblocking=True and lock is held)

    Raises:
        ValueError: If key is specified but not found in function signature

    Example:
        @synchronized_by_arg_value(index=0)
        def process_user(user_id: str) -> None:
            # Only one thread can process the same user_id at a time
            # But different user_ids can be processed concurrently
            pass

        @synchronized_by_arg_value(key='user_id')
        def process_user(user_id: str, data: dict) -> None:
            # More readable - synchronizes by user_id parameter
            pass

        @synchronized_by_arg_value(nonblocking=True)
        def try_update_cache(cache_key: str) -> bool:
            # Returns None if another thread is already updating this key
            pass
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        # Get function signature for key validation
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Validate key if provided
        if key and key not in param_names:
            raise ValueError(f"Parameter '{key}' not found in function signature. Available parameters: {param_names}")

        # Determine the index to use for key extraction
        target_index = param_names.index(key) if key else index

        # Shared state for all calls to this decorated function
        locks: dict[object, Lock] = {}  # Maps lock keys to their Lock objects
        usage_count: dict[object, int] = defaultdict(int)  # Reference counter for safe cleanup
        registry_lock = RLock()  # Protects locks and usage_count from race conditions

        def extract_key(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
            """Extract the locking key from function arguments."""
            if key and key in kwargs:
                return kwargs[key]
            if target_index >= len(args):
                raise IndexError(f"Argument index {target_index} out of range. Function called with {len(args)} arguments.")
            return args[target_index]

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            lock_key = extract_key(args, kwargs)

            # Step 1: Atomically get or create lock and increment usage count
            # We increment BEFORE acquiring the actual lock to prevent cleanup race conditions
            with registry_lock:
                if lock_key not in locks:
                    locks[lock_key] = Lock()
                lock = locks[lock_key]
                usage_count[lock_key] += 1  # This thread is now "using" this lock

            # Step 2: Try to acquire the actual lock (unified approach for blocking/nonblocking)
            # Using acquire(blocking=False) for nonblocking mode, acquire(blocking=True) for blocking
            acquired = lock.acquire(blocking=not nonblocking)

            if not acquired:
                # Failed to get lock in nonblocking mode - cleanup and return None
                # Must decrement usage count and potentially cleanup the lock
                with registry_lock:
                    usage_count[lock_key] -= 1
                    if usage_count[lock_key] == 0:
                        # No other threads are using this lock, safe to remove
                        locks.pop(lock_key, None)
                        usage_count.pop(lock_key, None)
                return None

            # Step 3: Successfully acquired lock - execute function and cleanup
            try:
                return func(*args, **kwargs)
            finally:
                # Always release the lock first
                lock.release()

                # Then cleanup usage tracking
                # This is in finally to ensure cleanup even if func() raises an exception
                with registry_lock:
                    usage_count[lock_key] -= 1
                    if usage_count[lock_key] == 0:
                        # Last thread using this lock - safe to remove from memory
                        # This prevents memory leaks for functions with many unique lock keys
                        locks.pop(lock_key, None)
                        usage_count.pop(lock_key, None)

        return wrapper

    return decorator


def synchronized[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that ensures all calls to a function are executed in synchronized manner.

    Creates a single lock for the function, guaranteeing that only one thread
    can execute the function at any time, regardless of arguments.

    Args:
        func: Function to synchronize

    Returns:
        Synchronized version of the function with the same signature

    Example:
        @synchronized
        def update_global_state() -> None:
            # Only one thread can execute this at a time
            global_counter += 1

        @synchronized
        def critical_section(data: dict) -> str:
            # All calls synchronized, even with different arguments
            return process_shared_resource(data)
    """
    lock = Lock()

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with lock:
            return func(*args, **kwargs)

    return wrapper
