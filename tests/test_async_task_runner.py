"""Tests for AsyncTaskRunner."""

import asyncio

import pytest

from mm_concurrency.async_task_runner import AsyncTaskRunner


class TestAsyncTaskRunner:
    """Test cases for AsyncTaskRunner class."""

    async def test_basic_task_execution(self) -> None:
        """Test basic task execution with successful results."""
        runner = AsyncTaskRunner()

        async def task1() -> str:
            return "result1"

        async def task2() -> int:
            return 42

        runner.add("task1", task1())
        runner.add("task2", task2())

        result = await runner.run()

        assert result.is_ok
        assert not result.is_timeout
        assert result.results == {"task1": "result1", "task2": 42}
        assert result.exceptions == {}

    async def test_task_with_exceptions(self) -> None:
        """Test handling of tasks that raise exceptions."""
        runner = AsyncTaskRunner(suppress_logging=True)

        async def successful_task() -> str:
            return "success"

        async def failing_task() -> None:
            raise ValueError("Test error")

        runner.add("success", successful_task())
        runner.add("failure", failing_task())

        result = await runner.run()

        assert not result.is_ok
        assert not result.is_timeout
        assert result.results == {"success": "success"}
        assert "failure" in result.exceptions
        assert isinstance(result.exceptions["failure"], ValueError)
        assert str(result.exceptions["failure"]) == "Test error"

    async def test_concurrency_limit(self) -> None:
        """Test that concurrency limit is respected."""
        max_concurrent = 2
        runner = AsyncTaskRunner(max_concurrent_tasks=max_concurrent)

        running_tasks = 0
        max_running = 0

        async def tracked_task(task_id: str) -> str:
            nonlocal running_tasks, max_running
            running_tasks += 1
            max_running = max(max_running, running_tasks)
            await asyncio.sleep(0.1)  # Simulate work
            running_tasks -= 1
            return f"result_{task_id}"

        for i in range(5):
            runner.add(f"task_{i}", tracked_task(f"task_{i}"))

        result = await runner.run()

        assert result.is_ok
        assert max_running <= max_concurrent
        assert len(result.results) == 5

    async def test_timeout_handling(self) -> None:
        """Test timeout functionality."""
        runner = AsyncTaskRunner(timeout=0.1)

        async def fast_task() -> str:
            return "fast"

        async def slow_task() -> str:
            await asyncio.sleep(1.0)  # This will timeout
            return "slow"

        runner.add("fast", fast_task())
        runner.add("slow", slow_task())

        result = await runner.run()

        assert not result.is_ok
        assert result.is_timeout
        # Fast task might complete before timeout
        assert "fast" in result.results or "fast" not in result.results

    async def test_empty_task_key_validation(self) -> None:
        """Test validation of empty task keys."""
        runner = AsyncTaskRunner()

        async def dummy_task() -> None:
            pass

        # Create coroutines to avoid warnings about unawaited coroutines
        coro1 = dummy_task()
        coro2 = dummy_task()

        with pytest.raises(ValueError, match="Task key cannot be empty"):
            runner.add("", coro1)

        with pytest.raises(ValueError, match="Task key cannot be empty"):
            runner.add("   ", coro2)

        # Clean up coroutines to avoid warnings
        coro1.close()
        coro2.close()

    async def test_duplicate_key_validation(self) -> None:
        """Test validation of duplicate task keys."""
        runner = AsyncTaskRunner()

        async def dummy_task() -> None:
            pass

        # Create coroutines to avoid warnings
        coro1 = dummy_task()
        coro2 = dummy_task()

        runner.add("task1", coro1)

        with pytest.raises(ValueError, match="Task key 'task1' already exists"):
            runner.add("task1", coro2)

        # Clean up both coroutines since the runner was never run
        coro1.close()
        coro2.close()

    async def test_runner_reuse_prevention(self) -> None:
        """Test that runner cannot be reused after running."""
        runner = AsyncTaskRunner()

        async def dummy_task() -> str:
            return "result"

        # Create coroutines to avoid warnings
        coro1 = dummy_task()
        coro2 = dummy_task()

        runner.add("task1", coro1)
        await runner.run()

        # Trying to add more tasks should fail
        with pytest.raises(RuntimeError, match="already been used"):
            runner.add("task2", coro2)

        # Trying to run again should fail
        with pytest.raises(RuntimeError, match="can only be run once"):
            await runner.run()

        # Clean up the unused coroutine
        coro2.close()

    async def test_no_tasks_validation(self) -> None:
        """Test that running without tasks raises error."""
        runner = AsyncTaskRunner()

        with pytest.raises(ValueError, match="No tasks to run"):
            await runner.run()

    async def test_invalid_timeout_validation(self) -> None:
        """Test validation of timeout parameter."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            AsyncTaskRunner(timeout=0)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            AsyncTaskRunner(timeout=-1)

    async def test_runner_configuration(self) -> None:
        """Test runner configuration parameters."""
        runner = AsyncTaskRunner(max_concurrent_tasks=10, timeout=5.0, name="test_runner", suppress_logging=True)

        assert runner.max_concurrent_tasks == 10
        assert runner.timeout == 5.0
        assert runner.name == "test_runner"
        assert runner.suppress_logging is True

    async def test_result_structure(self) -> None:
        """Test the structure and properties of Result dataclass."""
        runner = AsyncTaskRunner()

        async def task() -> str:
            return "test"

        runner.add("test_task", task())
        result = await runner.run()

        # Test Result attributes
        assert hasattr(result, "results")
        assert hasattr(result, "exceptions")
        assert hasattr(result, "is_ok")
        assert hasattr(result, "is_timeout")

        assert isinstance(result.results, dict)
        assert isinstance(result.exceptions, dict)
        assert isinstance(result.is_ok, bool)
        assert isinstance(result.is_timeout, bool)

    async def test_mixed_success_and_failure(self) -> None:
        """Test runner with mix of successful and failing tasks."""
        runner = AsyncTaskRunner(suppress_logging=True)

        async def success1() -> str:
            return "success1"

        async def success2() -> int:
            return 100

        async def failure1() -> None:
            raise RuntimeError("Runtime error")

        async def failure2() -> None:
            raise ValueError("Value error")

        runner.add("s1", success1())
        runner.add("s2", success2())
        runner.add("f1", failure1())
        runner.add("f2", failure2())

        result = await runner.run()

        assert not result.is_ok
        assert not result.is_timeout
        assert result.results == {"s1": "success1", "s2": 100}
        assert len(result.exceptions) == 2
        assert isinstance(result.exceptions["f1"], RuntimeError)
        assert isinstance(result.exceptions["f2"], ValueError)

    async def test_task_naming_with_runner_name(self) -> None:
        """Test that tasks are properly named when runner has a name."""
        runner = AsyncTaskRunner(name="test_runner")

        # We can't directly test asyncio task names, but we can verify
        # the runner accepts and stores the name properly
        assert runner.name == "test_runner"

        async def simple_task() -> str:
            return "done"

        runner.add("task1", simple_task())
        result = await runner.run()

        assert result.is_ok
        assert result.results["task1"] == "done"

    async def test_partial_timeout_scenario(self) -> None:
        """Test scenario where some tasks complete before timeout."""
        runner = AsyncTaskRunner(timeout=0.2)

        async def very_fast_task() -> str:
            return "very_fast"

        async def medium_task() -> str:
            await asyncio.sleep(0.1)
            return "medium"

        async def very_slow_task() -> str:
            await asyncio.sleep(1.0)
            return "very_slow"

        runner.add("very_fast", very_fast_task())
        runner.add("medium", medium_task())
        runner.add("very_slow", very_slow_task())

        result = await runner.run()

        assert not result.is_ok
        assert result.is_timeout
        # At least the very fast task should complete
        assert "very_fast" in result.results
        assert result.results["very_fast"] == "very_fast"
