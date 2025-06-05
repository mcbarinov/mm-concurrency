"""Tests for AsyncScheduler."""

import asyncio
from typing import Any

import pytest

from mm_concurrency.async_scheduler import AsyncScheduler


class TestAsyncScheduler:
    """Test cases for AsyncScheduler class."""

    async def test_basic_task_execution_and_lifecycle(self) -> None:
        """Test basic task execution, start/stop lifecycle, and statistics."""
        scheduler = AsyncScheduler("test_scheduler")
        execution_count = 0

        async def test_task() -> None:
            nonlocal execution_count
            execution_count += 1

        # Test adding task and initial state
        scheduler.add_task("task1", 0.05, test_task)
        assert not scheduler.is_running()
        assert len(scheduler.tasks) == 1
        assert scheduler.tasks["task1"].run_count == 0

        # Test start and execution
        scheduler.start()
        assert scheduler.is_running()

        await asyncio.sleep(0.15)  # Allow ~3 executions

        await scheduler.stop()
        assert not scheduler.is_running()

        # Give time for tasks to finish cleanup
        await asyncio.sleep(0.05)

        # Verify task was executed multiple times
        assert execution_count >= 2
        assert scheduler.tasks["task1"].run_count >= 2
        assert scheduler.tasks["task1"].last_run is not None
        assert not scheduler.tasks["task1"].running

    async def test_multiple_tasks_parallel_execution(self) -> None:
        """Test that multiple tasks run in parallel with different intervals."""
        scheduler = AsyncScheduler()
        task1_count = 0
        task2_count = 0

        async def fast_task() -> None:
            nonlocal task1_count
            task1_count += 1

        async def slow_task() -> None:
            nonlocal task2_count
            task2_count += 1

        scheduler.add_task("fast", 0.03, fast_task)
        scheduler.add_task("slow", 0.08, slow_task)

        scheduler.start()
        await asyncio.sleep(0.2)
        await scheduler.stop()

        # Fast task should execute more times than slow task
        assert task1_count > task2_count
        assert scheduler.tasks["fast"].run_count > scheduler.tasks["slow"].run_count

    async def test_task_arguments_and_kwargs(self) -> None:
        """Test that tasks receive correct arguments and keyword arguments."""
        scheduler = AsyncScheduler()
        received_args: list[Any] = []
        received_kwargs: list[dict[str, Any]] = []

        async def task_with_args(arg1: str, arg2: int, keyword: str = "default") -> None:
            received_args.append((arg1, arg2))
            received_kwargs.append({"keyword": keyword})

        scheduler.add_task("args_task", 0.1, task_with_args, ("test", 42), {"keyword": "value"})

        scheduler.start()
        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert len(received_args) >= 1
        assert received_args[0] == ("test", 42)
        assert received_kwargs[0] == {"keyword": "value"}

    async def test_error_handling_and_error_count(self) -> None:
        """Test that task errors are handled and counted properly."""
        scheduler = AsyncScheduler()
        error_count = 0
        success_count = 0

        async def failing_task() -> None:
            nonlocal error_count, success_count
            error_count += 1
            if error_count <= 2:
                raise ValueError(f"Error {error_count}")
            success_count += 1

        scheduler.add_task("failing", 0.05, failing_task)

        scheduler.start()
        await asyncio.sleep(0.2)  # Allow several executions
        await scheduler.stop()

        task_info = scheduler.tasks["failing"]
        assert task_info.error_count == 2  # First 2 calls failed
        assert task_info.run_count > 2  # More than 2 total executions
        assert success_count > 0  # Some successful executions after failures

    async def test_duplicate_task_id_validation(self) -> None:
        """Test that adding duplicate task IDs raises ValueError."""
        scheduler = AsyncScheduler()

        async def dummy_task() -> None:
            pass

        scheduler.add_task("task1", 1.0, dummy_task)

        with pytest.raises(ValueError, match="Task with id task1 already exists"):
            scheduler.add_task("task1", 2.0, dummy_task)

    async def test_start_stop_edge_cases(self) -> None:
        """Test edge cases for start/stop operations."""
        scheduler = AsyncScheduler()

        async def dummy_task() -> None:
            pass

        scheduler.add_task("task1", 1.0, dummy_task)

        # Test multiple starts
        scheduler.start()
        assert scheduler.is_running()

        scheduler.start()  # Should be no-op
        assert scheduler.is_running()

        # Test multiple stops
        await scheduler.stop()
        assert not scheduler.is_running()

        await scheduler.stop()  # Should be no-op
        assert not scheduler.is_running()

    async def test_clear_tasks_functionality(self) -> None:
        """Test clearing tasks and validation when running."""
        scheduler = AsyncScheduler()

        async def dummy_task() -> None:
            await asyncio.sleep(0.01)

        scheduler.add_task("task1", 0.1, dummy_task)
        scheduler.add_task("task2", 0.1, dummy_task)
        assert len(scheduler.tasks) == 2

        # Cannot clear while running
        scheduler.start()
        scheduler.clear_tasks()
        assert len(scheduler.tasks) == 2  # Should still be there

        await scheduler.stop()

        # Can clear when stopped
        scheduler.clear_tasks()
        assert len(scheduler.tasks) == 0

    async def test_scheduler_name_propagation(self) -> None:
        """Test that scheduler name is properly used in task naming."""
        scheduler = AsyncScheduler("custom_name")

        async def dummy_task() -> None:
            await asyncio.sleep(0.01)

        scheduler.add_task("test_task", 0.1, dummy_task)
        scheduler.start()

        # Give time for task to start
        await asyncio.sleep(0.05)

        # Verify task is running
        assert scheduler.tasks["test_task"].running

        await scheduler.stop()

        # Give time for tasks to finish cleanup
        await asyncio.sleep(0.05)

        # Verify task is running
        assert scheduler.tasks["test_task"].running is False

    async def test_empty_scheduler_start_stop(self) -> None:
        """Test starting/stopping scheduler with no tasks."""
        scheduler = AsyncScheduler()

        scheduler.start()
        assert scheduler.is_running()

        await asyncio.sleep(0.05)

        await scheduler.stop()
        assert not scheduler.is_running()
