"""Concurrent async task execution with result collection and error handling."""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

type AsyncFunc = Callable[..., Coroutine[Any, Any, object]]
type Args = tuple[object, ...]
type Kwargs = dict[str, object]
type TaskKey = str
type TaskResult = object


class AsyncTaskRunner:
    """Execute multiple async tasks concurrently and collect results by key.

    Manages asyncio tasks to run coroutines concurrently with configurable
    concurrency limit, tracking results and exceptions for each task by its unique key.

    Example:
        runner = AsyncTaskRunner(max_concurrent_tasks=3)
        runner.add("task1", fetch_data_async, ("url1",))
        runner.add("task2", process_file_async, ("file.txt",))
        await runner.run()

        if runner.has_errors:
            print(f"Failed: {runner.exceptions}")
        print(f"Results: {runner.results}")
    """

    def __init__(self, max_concurrent_tasks: int = 5, timeout: float | None = None) -> None:
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout = timeout
        self.tasks: list[AsyncTaskRunner.Task] = []
        self.exceptions: dict[TaskKey, Exception] = {}  # Exceptions for failed tasks by key
        self.has_errors = False  # True if any task failed or timed out
        self.has_timeout = False  # True if execution timed out
        self.results: dict[TaskKey, TaskResult] = {}  # Results for successful tasks by key

    @dataclass
    class Task:
        key: TaskKey
        func: AsyncFunc
        args: Args
        kwargs: Kwargs

    def add(self, key: TaskKey, func: AsyncFunc, args: Args = (), kwargs: Kwargs | None = None) -> None:
        """Add an async task to be executed.

        Args:
            key: Unique identifier for this task
            func: Async function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
        """
        if kwargs is None:
            kwargs = {}
        self.tasks.append(AsyncTaskRunner.Task(key, func, args, kwargs))

    async def run(self) -> None:
        """Execute all added async tasks concurrently.

        Results are stored in self.results and exceptions in self.exceptions.
        Check self.has_errors and self.has_timeout for execution status.
        """
        if not self.tasks:
            return

        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def _run_task_with_semaphore(task: AsyncTaskRunner.Task) -> None:
            """Run a single task with semaphore protection."""
            async with semaphore:
                try:
                    result = await task.func(*task.args, **task.kwargs)
                    self.results[task.key] = result
                except Exception as err:
                    self.has_errors = True
                    self.exceptions[task.key] = err

        try:
            # Create tasks with semaphore protection
            async_tasks: dict[asyncio.Task[None], TaskKey] = {
                asyncio.create_task(_run_task_with_semaphore(task), name=task.key): task.key for task in self.tasks
            }

            # Use asyncio.wait to handle individual task failures
            done, pending = await asyncio.wait(async_tasks.keys(), timeout=self.timeout, return_when=asyncio.ALL_COMPLETED)

            # Handle timeout - cancel pending tasks
            if pending:
                self.has_errors = True
                self.has_timeout = True
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        except TimeoutError:
            self.has_errors = True
            self.has_timeout = True
