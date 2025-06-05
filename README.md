# mm-concurrency

A Python library for elegant concurrent and asynchronous task execution with built-in error handling, result collection, and synchronization utilities.

## Features

- **Task Runners**: Execute multiple tasks concurrently with configurable limits and timeout support
- **Synchronization Decorators**: Coordinate access to shared resources in both sync and async contexts
- **Error Handling**: Comprehensive exception tracking and result collection
- **Type Safe**: Full type annotations with mypy support
- **Modern Python**: Built for Python 3.13+ with latest async/await patterns


## Quick Start

### Async Task Runner

Execute multiple async tasks concurrently with automatic resource management:

```python
import asyncio
from mm_concurrency import AsyncTaskRunner

async def fetch_data(url: str) -> dict:
    # Your async data fetching logic
    await asyncio.sleep(1)
    return {"url": url, "data": "some data"}

async def main():
    # Create runner with concurrency limit and timeout
    runner = AsyncTaskRunner(max_concurrent_tasks=3, timeout=10.0)

    # Add tasks with unique keys
    runner.add("user1", fetch_data("https://api.example.com/user/1"))
    runner.add("user2", fetch_data("https://api.example.com/user/2"))
    runner.add("user3", fetch_data("https://api.example.com/user/3"))

    # Execute all tasks
    result = await runner.run()

    if result.is_ok:
        print(f"All tasks completed: {result.results}")
    else:
        print(f"Some tasks failed: {result.exceptions}")

asyncio.run(main())
```

### Sync Task Runner

For CPU-bound or blocking I/O operations:

```python
from mm_concurrency import TaskRunner
import requests

def fetch_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()

# Create and run tasks
runner = TaskRunner(max_concurrent_tasks=3, timeout=30.0)
runner.add("api1", fetch_data, ("https://api.example.com/data1",))
runner.add("api2", fetch_data, ("https://api.example.com/data2",))

result = runner.run()
print(f"Results: {result.results}")
```

### Async Synchronization

Coordinate concurrent access to shared resources:

```python
from mm_concurrency import async_synchronized, async_synchronized_by_arg_value

# Synchronize all calls to a function
@async_synchronized
async def update_global_state() -> None:
    # Only one coroutine can execute this at a time
    global_counter += 1
    await asyncio.sleep(0.1)

# Synchronize by argument value
@async_synchronized_by_arg_value(key='user_id')
async def process_user_data(user_id: str, data: dict) -> dict:
    # Only one coroutine per user_id, but different users can run concurrently
    await asyncio.sleep(1)
    return {"user_id": user_id, "processed": True}

# Non-blocking synchronization
@async_synchronized_by_arg_value(key='resource', nonblocking=True)
async def try_update_cache(resource: str) -> str | None:
    # Returns None if another coroutine is already processing this resource
    await expensive_operation(resource)
    return "updated"
```

### Thread Synchronization

Same patterns for thread-based concurrency:

```python
from mm_concurrency import synchronized, synchronized_by_arg_value

@synchronized
def thread_safe_operation() -> None:
    # Only one thread at a time
    global shared_state
    shared_state += 1

@synchronized_by_arg_value(key='user_id')
def process_user(user_id: str, data: dict) -> dict:
    # Per-user synchronization
    return update_user_data(user_id, data)
```

## API Reference

### AsyncTaskRunner

Execute async tasks concurrently with resource limits:

```python
runner = AsyncTaskRunner(
    max_concurrent_tasks=5,  # Limit concurrent execution
    timeout=30.0,           # Overall timeout in seconds
    name="my_runner",       # Optional name for debugging
    suppress_logging=False  # Control exception logging
)

runner.add(key="task1", awaitable=my_coroutine())
result = await runner.run()

# Result object contains:
# result.results: dict[str, Any] - successful results by key
# result.exceptions: dict[str, Exception] - exceptions by key
# result.is_ok: bool - True if no errors or timeouts
# result.is_timeout: bool - True if timeout occurred
```

### TaskRunner

Execute sync tasks in thread pool:

```python
runner = TaskRunner(
    max_concurrent_tasks=5,
    timeout=30.0,
    name="my_runner",
    suppress_logging=False
)

runner.add(key="task1", func=my_function, args=(arg1,), kwargs={"key": "value"})
result = runner.run()  # Same Result structure as AsyncTaskRunner
```

### Synchronization Decorators

#### Basic Synchronization

- `@synchronized` - Synchronize all calls to a function (threads)
- `@async_synchronized` - Synchronize all calls to an async function (coroutines)

#### By-Value Synchronization

- `@synchronized_by_arg_value(index=0, key=None, nonblocking=False)` - Thread synchronization by argument
- `@async_synchronized_by_arg_value(index=0, key=None, nonblocking=False)` - Coroutine synchronization by argument

Parameters:
- `index`: Position of argument to use as lock key (default: 0)
- `key`: Name of parameter to use as lock key (overrides index)
- `nonblocking`: Return None immediately if lock is held (default: False)

## Advanced Examples

### Error Handling and Timeouts

```python
async def main():
    runner = AsyncTaskRunner(max_concurrent_tasks=2, timeout=5.0)

    runner.add("fast", quick_task())
    runner.add("slow", slow_task())
    runner.add("failing", failing_task())

    result = await runner.run()

    # Handle different outcomes
    if result.is_timeout:
        print("Some tasks timed out")

    for key, exception in result.exceptions.items():
        print(f"Task {key} failed: {exception}")

    for key, value in result.results.items():
        print(f"Task {key} succeeded: {value}")
```

### Dynamic Synchronization

```python
# Synchronize by user ID - each user gets their own lock
@async_synchronized_by_arg_value(key='user_id')
async def update_user_profile(user_id: str, profile_data: dict) -> None:
    # Multiple users can update concurrently, but each user is serialized
    await database.update_user(user_id, profile_data)

# Non-blocking cache updates
@async_synchronized_by_arg_value(key='cache_key', nonblocking=True)
async def refresh_cache(cache_key: str) -> dict | None:
    # If cache is being refreshed, return None instead of waiting
    if cache_key in cache:
        return cache[cache_key]

    new_data = await fetch_expensive_data(cache_key)
    cache[cache_key] = new_data
    return new_data
```
