"""Concurrent async task execution with result collection and error handling."""

import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import anyio

type TaskKey = str
type TaskResult = Any

logger = logging.getLogger(__name__)


class AsyncTaskRunner:
    """Execute multiple async tasks concurrently and collect results by key.

    Manages anyio task groups to run coroutines concurrently with configurable
    concurrency limit, tracking results and exceptions for each task by its unique key.

    Note: This runner is designed for one-time use. Create a new instance for each batch of tasks.

    Example:
        runner = AsyncTaskRunner(max_concurrent_tasks=3, timeout=10.5, name="data_fetcher")
        runner.add("task1", fetch_data_async("url1"))
        runner.add("task2", process_file_async("file.txt"))
        result = await runner.run()

        if not result.is_ok:
            print(f"Failed: {result.exceptions}")
        print(f"Results: {result.results}")
    """

    @dataclass
    class Result:
        results: dict[TaskKey, TaskResult]  # Maps task_key to result
        exceptions: dict[TaskKey, Exception]  # Maps task_key to exception (if any)
        is_ok: bool  # True if no exception and no timeout occurred
        is_timeout: bool  # True if at least one task was cancelled due to timeout

    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        timeout: float | None = None,
        name: str | None = None,
        suppress_logging: bool = False,
    ) -> None:
        """Initialize AsyncTaskRunner.

        Args:
            max_concurrent_tasks: Maximum number of tasks that can run concurrently
            timeout: Optional overall timeout in seconds for running all tasks
            name: Optional name for the runner (useful for debugging)
            suppress_logging: If True, suppresses logging for task exceptions

        Raises:
            ValueError: If timeout is not positive
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be positive if specified")

        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout = timeout
        self.name = name
        self.suppress_logging = suppress_logging
        self.tasks: list[AsyncTaskRunner.Task] = []
        self._task_keys: set[TaskKey] = set()
        self._was_run = False

    @dataclass
    class Task:
        key: TaskKey
        awaitable: Awaitable[Any]

    def add(self, key: TaskKey, awaitable: Awaitable[Any]) -> None:
        """Add an async task to be executed.

        Args:
            key: Unique identifier for this task
            awaitable: Awaitable object (coroutine) to execute

        Raises:
            RuntimeError: If the runner has already been used
            ValueError: If key is empty or already exists
        """
        if self._was_run:
            raise RuntimeError("This AsyncTaskRunner has already been used. Create a new instance for new tasks.")

        if not key or not key.strip():
            raise ValueError("Task key cannot be empty")

        if key in self._task_keys:
            raise ValueError(f"Task key '{key}' already exists")

        self._task_keys.add(key)
        self.tasks.append(AsyncTaskRunner.Task(key, awaitable))

    async def run(self) -> "AsyncTaskRunner.Result":
        """Execute all added async tasks concurrently.

        Returns AsyncTaskRunner.Result containing task results, exceptions,
        and flags indicating overall status.

        Raises:
            RuntimeError: If the runner has already been used
            ValueError: If no tasks have been added
        """
        if self._was_run:
            raise RuntimeError("This AsyncTaskRunner instance can only be run once. Create a new instance for new tasks.")

        self._was_run = True

        if not self.tasks:
            raise ValueError("No tasks to run. Add tasks using add() method before calling run()")

        results: dict[TaskKey, TaskResult] = {}
        exceptions: dict[TaskKey, Exception] = {}
        is_timeout = False

        async def _run_task_with_limiter(task: AsyncTaskRunner.Task, limiter: anyio.CapacityLimiter) -> None:
            """Run a single task with capacity limiter protection."""
            async with limiter:
                try:
                    result = await task.awaitable
                    results[task.key] = result
                except Exception as err:
                    if not self.suppress_logging:
                        logger.exception("Task raised an exception", extra={"task_key": task.key})
                    exceptions[task.key] = err

        def _task_name(task_key: TaskKey) -> str:
            return f"{self.name}-{task_key}" if self.name else task_key

        try:
            # Use anyio's fail_after for timeout support if specified
            if self.timeout is not None:
                with anyio.fail_after(self.timeout):
                    # TaskGroup automatically handles task lifecycle and cancellation
                    async with anyio.create_task_group() as tg:
                        limiter = anyio.CapacityLimiter(self.max_concurrent_tasks)

                        for task in self.tasks:
                            tg.start_soon(_run_task_with_limiter, task, limiter)
            else:
                # TaskGroup automatically handles task lifecycle and cancellation
                async with anyio.create_task_group() as tg:
                    limiter = anyio.CapacityLimiter(self.max_concurrent_tasks)

                    for task in self.tasks:
                        tg.start_soon(_run_task_with_limiter, task, limiter)

        except TimeoutError:
            is_timeout = True
        except anyio.get_cancelled_exc_class():
            is_timeout = True

        is_ok = not exceptions and not is_timeout
        return AsyncTaskRunner.Result(results=results, exceptions=exceptions, is_ok=is_ok, is_timeout=is_timeout)
