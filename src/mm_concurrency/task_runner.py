"""Concurrent task execution with result collection and error handling."""

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass

type Func = Callable[..., object]
type Args = tuple[object, ...]
type Kwargs = dict[str, object]
type TaskKey = str
type TaskResult = object


class TaskRunner:
    """Execute multiple tasks concurrently and collect results by key.

    Manages a ThreadPoolExecutor to run tasks concurrently, tracking
    results and exceptions for each task by its unique key.

    Example:
        runner = TaskRunner(max_workers=5)
        runner.add("task1", fetch_data, ("url1",))
        runner.add("task2", process_file, ("file.txt",))
        runner.run()

        if runner.has_errors:
            print(f"Failed: {runner.exceptions}")
        print(f"Results: {runner.results}")
    """

    def __init__(self, max_workers: int = 5, timeout: float | None = None, thread_prefix: str = "task_runner") -> None:
        self.max_workers = max_workers
        self.timeout = timeout
        self.thread_prefix = thread_prefix
        self.tasks: list[TaskRunner.Task] = []
        self.exceptions: dict[TaskKey, Exception] = {}  # Exceptions for failed tasks by key
        self.has_errors = False  # True if any task failed or timed out
        self.has_timeout = False  # True if execution timed out
        self.results: dict[TaskKey, TaskResult] = {}  # Results for successful tasks by key

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
        """
        if kwargs is None:
            kwargs = {}
        self.tasks.append(TaskRunner.Task(key, func, args, kwargs))

    def run(self) -> None:
        """Execute all added tasks concurrently.

        Results are stored in self.results and exceptions in self.exceptions.
        Check self.has_errors and self.has_timeout for execution status.
        """
        with concurrent.futures.ThreadPoolExecutor(self.max_workers, thread_name_prefix=self.thread_prefix) as executor:
            future_to_key = {executor.submit(task.func, *task.args, **task.kwargs): task.key for task in self.tasks}
            try:
                result_map = concurrent.futures.as_completed(future_to_key, timeout=self.timeout)
                for future in result_map:
                    key = future_to_key[future]
                    try:
                        self.results[key] = future.result()
                    except Exception as err:
                        self.has_errors = True
                        self.exceptions[key] = err
            except concurrent.futures.TimeoutError:
                self.has_errors = True
                self.has_timeout = True
