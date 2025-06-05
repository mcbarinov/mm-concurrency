import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mm_concurrency.utils import utc_now

type AsyncFunc = Callable[..., Awaitable[object]]
type Args = tuple[object, ...]
type Kwargs = dict[str, object]

logger = logging.getLogger(__name__)


class AsyncScheduler:
    """
    A scheduler for running async tasks at fixed intervals.

    Each task runs on its own schedule and waits for the specified interval
    between executions.

    Design Notes:
        - No context manager needed - explicit start/stop is preferred
        - No individual task removal - clear_tasks() is sufficient
        - Must be compatible with uvloop
    """

    @dataclass
    class TaskInfo:
        """Information about a scheduled task."""

        task_id: str
        interval: float
        func: AsyncFunc
        args: Args = ()
        kwargs: Kwargs = field(default_factory=dict)
        run_count: int = 0
        error_count: int = 0
        last_run: datetime | None = None
        running: bool = False

    def __init__(self, name: str = "AsyncScheduler") -> None:
        """Initialize the async scheduler."""
        self.tasks: dict[str, AsyncScheduler.TaskInfo] = {}
        self._running: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._main_task: asyncio.Task[Any] | None = None
        self._name = name

    def add_task(self, task_id: str, interval: float, func: AsyncFunc, args: Args = (), kwargs: Kwargs | None = None) -> None:
        """
        Register a new task with the scheduler.

        Args:
            task_id: Unique identifier for the task
            interval: Time in seconds between task executions
            func: Async function to execute
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function

        Raises:
            ValueError: If a task with the same ID already exists
        """
        if kwargs is None:
            kwargs = {}
        if task_id in self.tasks:
            raise ValueError(f"Task with id {task_id} already exists")
        self.tasks[task_id] = AsyncScheduler.TaskInfo(task_id=task_id, interval=interval, func=func, args=args, kwargs=kwargs)

    async def _run_task(self, task_id: str) -> None:
        """
        Internal loop for running a single task repeatedly.

        Args:
            task_id: ID of the task to run
        """
        task = self.tasks[task_id]
        task.running = True

        elapsed = 0.0
        try:
            while self._running:
                task.last_run = utc_now()
                task.run_count += 1
                try:
                    await task.func(*task.args, **task.kwargs)
                except Exception:
                    task.error_count += 1
                    logger.exception("Error in task", extra={"task_id": task_id, "error_count": task.error_count})

                # Calculate elapsed time and sleep if needed
                elapsed = (utc_now() - task.last_run).total_seconds()
                sleep_time = max(0.0, task.interval - elapsed)
                if sleep_time > 0:
                    try:
                        await asyncio.sleep(sleep_time)
                    except asyncio.CancelledError:
                        break
        finally:
            task.running = False
            logger.debug("Finished task", extra={"task_id": task_id, "elapsed": elapsed})

    async def _start_all_tasks(self) -> None:
        """Starts all tasks concurrently using asyncio tasks."""
        self._tasks = []

        for task_id in self.tasks:
            task = asyncio.create_task(self._run_task(task_id), name=self._name + "-" + task_id)
            self._tasks.append(task)

        try:
            # Keep the main task alive while the scheduler is running
            while self._running:  # noqa: ASYNC110
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.debug("Cancelled all tasks")
        finally:
            # Cancel all running tasks when we exit
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to finish
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []

    def start(self) -> None:
        """
        Start the scheduler.

        Creates tasks in the current event loop for each registered task.
        """
        if self._running:
            logger.warning("AsyncScheduler already running")
            return

        self._running = True
        logger.debug("starting")
        self._main_task = asyncio.create_task(self._start_all_tasks())

    async def stop(self) -> None:
        """
        Stop the scheduler gracefully.

        Sets running flag to False and waits for all tasks to complete.
        After 5 seconds, forcefully cancels any remaining tasks.
        """
        if not self._running:
            logger.warning("not running")
            return

        logger.debug("stopping")
        self._running = False

        if self._main_task and not self._main_task.done():
            try:
                # Wait up to 5 seconds for graceful shutdown
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Graceful shutdown timeout, force cancelling tasks")
                self._main_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._main_task

        logger.debug("stopped")

    def is_running(self) -> bool:
        """
        Check if the scheduler is currently running.

        Returns:
            True if the scheduler is running, False otherwise
        """
        return self._running

    def clear_tasks(self) -> None:
        """Clear all tasks from the scheduler."""
        if self._running:
            logger.warning("Cannot clear tasks while scheduler is running")
            return
        self.tasks.clear()
        logger.debug("cleared tasks")
