"""Concurrent task execution with result collection and error handling."""

import concurrent.futures
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

type Func = Callable[..., Any]
type Args = tuple[Any, ...]
type Kwargs = dict[str, Any]
type TaskKey = str
type TaskResult = Any

logger = logging.getLogger(__name__)


class TaskRunner:
    """Execute multiple tasks concurrently and collect results by key.

    Manages a ThreadPoolExecutor to run tasks concurrently with configurable
    concurrency limit, tracking results and exceptions for each task by its unique key.

    Note: This runner is designed for one-time use. Create a new instance for each batch of tasks.

    Example:
        runner = TaskRunner(max_concurrent_tasks=3, timeout=10.5, name="data_processor")
        runner.add("task1", fetch_data, ("url1",))
        runner.add("task2", process_file, ("file.txt",))
        result = runner.run()

        if not result.is_ok:
            print(f"Failed: {result.exceptions}")
        print(f"Results: {result.results}")
    """

    @dataclass
    class Result:
        results: dict[TaskKey, TaskResult]  # Maps task_key to result
        exceptions: dict[TaskKey, Exception]  # Maps task_key to exception (if any)
        is_ok: bool  # True if no exception and no timeout occurred
        is_timeout: bool  # True if execution was cancelled due to timeout

    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        timeout: float | None = None,
        name: str | None = None,
        suppress_logging: bool = False,
    ) -> None:
        """Initialize TaskRunner.

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
        self.tasks: list[TaskRunner.Task] = []
        self._task_keys: set[TaskKey] = set()
        self._was_run = False

    @dataclass
    class Task:
        key: TaskKey
        func: Func
        args: Args
        kwargs: Kwargs

    def add(self, key: TaskKey, func: Func, args: Args = (), kwargs: Kwargs | None = None) -> None:
        """Add a task to be executed.

        Args:
            key: Unique identifier for this task
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function

        Raises:
            RuntimeError: If the runner has already been used
            ValueError: If key is empty or already exists
        """
        if self._was_run:
            raise RuntimeError("This TaskRunner has already been used. Create a new instance for new tasks.")

        if not key or not key.strip():
            raise ValueError("Task key cannot be empty")

        if key in self._task_keys:
            raise ValueError(f"Task key '{key}' already exists")

        if kwargs is None:
            kwargs = {}

        self._task_keys.add(key)
        self.tasks.append(TaskRunner.Task(key, func, args, kwargs))

    def run(self) -> "TaskRunner.Result":
        """Execute all added tasks concurrently.

        Returns TaskRunner.Result containing task results, exceptions,
        and flags indicating overall status.

        Raises:
            RuntimeError: If the runner has already been used
            ValueError: If no tasks have been added
        """
        if self._was_run:
            raise RuntimeError("This TaskRunner instance can only be run once. Create a new instance for new tasks.")

        self._was_run = True

        if not self.tasks:
            raise ValueError("No tasks to run. Add tasks using add() method before calling run()")

        results: dict[TaskKey, TaskResult] = {}
        exceptions: dict[TaskKey, Exception] = {}
        is_timeout = False

        thread_name_prefix = f"{self.name}_task_runner" if self.name else "task_runner"

        with concurrent.futures.ThreadPoolExecutor(self.max_concurrent_tasks, thread_name_prefix=thread_name_prefix) as executor:
            future_to_key = {executor.submit(task.func, *task.args, **task.kwargs): task.key for task in self.tasks}
            try:
                result_map = concurrent.futures.as_completed(future_to_key, timeout=self.timeout)
                for future in result_map:
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except Exception as err:
                        if not self.suppress_logging:
                            logger.exception("Task raised an exception", extra={"task_key": key})
                        exceptions[key] = err
            except concurrent.futures.TimeoutError:
                is_timeout = True

        is_ok = not exceptions and not is_timeout
        return TaskRunner.Result(results=results, exceptions=exceptions, is_ok=is_ok, is_timeout=is_timeout)
