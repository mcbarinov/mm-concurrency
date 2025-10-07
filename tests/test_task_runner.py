"""Tests for TaskRunner class."""

import logging
import threading
import time
from typing import Any

import pytest

from mm_concurrency import TaskRunner


class TestTaskRunner:
    """Tests for the TaskRunner class."""

    def test_successful_execution(self) -> None:
        """Test basic successful execution of multiple tasks."""

        def task1() -> str:
            return "result1"

        def task2(value: int) -> int:
            return value * 2

        def task3(prefix: str, suffix: str) -> str:
            return f"{prefix}_{suffix}"

        runner = TaskRunner(max_concurrent_tasks=2)
        runner.add("task1", task1)
        runner.add("task2", task2, (10,))
        runner.add("task3", task3, ("hello", "world"))

        result = runner.run()

        assert result.is_ok
        assert not result.is_timeout
        assert len(result.results) == 3
        assert len(result.exceptions) == 0
        assert result.results["task1"] == "result1"
        assert result.results["task2"] == 20
        assert result.results["task3"] == "hello_world"

    def test_task_with_kwargs(self) -> None:
        """Test task execution with keyword arguments."""

        def task_with_kwargs(value: int, multiplier: int = 2, prefix: str = "result") -> str:
            return f"{prefix}_{value * multiplier}"

        runner = TaskRunner()
        runner.add("task1", task_with_kwargs, (5,), {"multiplier": 3})
        runner.add("task2", task_with_kwargs, (10,), {"prefix": "output"})

        result = runner.run()

        assert result.is_ok
        assert result.results["task1"] == "result_15"
        assert result.results["task2"] == "output_20"

    def test_exception_handling(self) -> None:
        """Test that exceptions are properly captured and do not stop other tasks."""

        def failing_task() -> str:
            raise ValueError("Task failed")

        def successful_task() -> str:
            return "success"

        runner = TaskRunner()
        runner.add("fail", failing_task)
        runner.add("success", successful_task)

        result = runner.run()

        assert not result.is_ok
        assert not result.is_timeout
        assert len(result.results) == 1
        assert len(result.exceptions) == 1
        assert result.results["success"] == "success"
        assert isinstance(result.exceptions["fail"], ValueError)
        assert str(result.exceptions["fail"]) == "Task failed"

    def test_mixed_success_and_failure(self) -> None:
        """Test execution with mix of successful and failing tasks."""

        def task1() -> int:
            return 42

        def task2() -> str:
            raise RuntimeError("Runtime error")

        def task3() -> str:
            time.sleep(0.01)  # Small delay
            return "delayed_result"

        def task4() -> None:
            raise ValueError("Value error")

        runner = TaskRunner(max_concurrent_tasks=2)
        runner.add("success1", task1)
        runner.add("error1", task2)
        runner.add("success2", task3)
        runner.add("error2", task4)

        result = runner.run()

        assert not result.is_ok
        assert not result.is_timeout
        assert len(result.results) == 2
        assert len(result.exceptions) == 2
        assert result.results["success1"] == 42
        assert result.results["success2"] == "delayed_result"
        assert isinstance(result.exceptions["error1"], RuntimeError)
        assert isinstance(result.exceptions["error2"], ValueError)

    def test_concurrent_execution(self) -> None:
        """Test that tasks actually run concurrently."""
        execution_order = []

        def slow_task(name: str) -> str:
            execution_order.append(f"start_{name}")
            time.sleep(0.1)
            execution_order.append(f"end_{name}")
            return f"result_{name}"

        start_time = time.time()

        runner = TaskRunner(max_concurrent_tasks=3)
        runner.add("task1", slow_task, ("task1",))
        runner.add("task2", slow_task, ("task2",))
        runner.add("task3", slow_task, ("task3",))

        result = runner.run()
        end_time = time.time()

        # With 3 concurrent tasks, total time should be ~0.1s, not ~0.3s
        assert end_time - start_time < 0.2

        assert result.is_ok
        assert len(result.results) == 3

        # All tasks should start before any end (proving concurrency)
        start_count = sum(1 for entry in execution_order if entry.startswith("start_"))
        first_end_index = next(i for i, entry in enumerate(execution_order) if entry.startswith("end_"))
        assert first_end_index >= start_count

    def test_concurrency_limit(self) -> None:
        """Test that concurrency limit is respected."""
        active_tasks = []
        max_concurrent = 0

        def monitored_task(task_id: str) -> str:
            active_tasks.append(task_id)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(active_tasks))
            time.sleep(0.05)
            active_tasks.remove(task_id)
            return f"done_{task_id}"

        runner = TaskRunner(max_concurrent_tasks=2)
        for i in range(5):
            runner.add(f"task{i}", monitored_task, (f"task{i}",))

        result = runner.run()

        assert result.is_ok
        assert max_concurrent <= 2
        assert len(result.results) == 5

    def test_timeout_handling(self) -> None:
        """Test timeout functionality."""

        def slow_task() -> str:
            time.sleep(0.2)
            return "should_not_complete"

        runner = TaskRunner(timeout=0.1)
        runner.add("slow", slow_task)

        result = runner.run()

        assert not result.is_ok
        assert result.is_timeout
        assert len(result.results) == 0
        assert len(result.exceptions) == 0

    def test_no_tasks(self) -> None:
        """Test error when trying to run with no tasks."""
        runner = TaskRunner()

        with pytest.raises(ValueError, match="No tasks to run"):
            runner.run()

    def test_duplicate_keys(self) -> None:
        """Test error when adding duplicate task keys."""

        def simple_task() -> str:
            return "result"

        runner = TaskRunner()
        runner.add("task1", simple_task)

        with pytest.raises(ValueError, match="Task key 'task1' already exists"):
            runner.add("task1", simple_task)

    def test_empty_key_validation(self) -> None:
        """Test validation of task keys."""

        def simple_task() -> str:
            return "result"

        runner = TaskRunner()

        with pytest.raises(ValueError, match="Task key cannot be empty"):
            runner.add("", simple_task)

        with pytest.raises(ValueError, match="Task key cannot be empty"):
            runner.add("   ", simple_task)

    def test_one_time_use_protection(self) -> None:
        """Test that TaskRunner can only be used once."""

        def simple_task() -> str:
            return "result"

        runner = TaskRunner()
        runner.add("task1", simple_task)

        # First run should work
        result1 = runner.run()
        assert result1.is_ok

        # Second run should raise error
        with pytest.raises(RuntimeError, match="This TaskRunner instance can only be run once"):
            runner.run()

        # Adding tasks after run should also raise error
        with pytest.raises(RuntimeError, match="This TaskRunner has already been used"):
            runner.add("task2", simple_task)

    def test_constructor_validation(self) -> None:
        """Test validation of constructor parameters."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            TaskRunner(timeout=0)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            TaskRunner(timeout=-1)

        # Valid timeouts should work
        TaskRunner(timeout=1.0)
        TaskRunner(timeout=0.1)

    def test_task_return_types(self) -> None:
        """Test that various return types are handled correctly."""

        def return_none() -> None:
            return None

        def return_list() -> list[int]:
            return [1, 2, 3]

        def return_dict() -> dict[str, Any]:
            return {"key": "value", "number": 42}

        runner = TaskRunner()
        runner.add("none", return_none)
        runner.add("list", return_list)
        runner.add("dict", return_dict)

        result = runner.run()

        assert result.is_ok
        assert result.results["none"] is None
        assert result.results["list"] == [1, 2, 3]
        assert result.results["dict"] == {"key": "value", "number": 42}

    def test_thread_prefix_configuration(self) -> None:
        """Test that thread name prefix is configurable."""
        executed_thread_names = []

        def capture_thread_name() -> str:
            executed_thread_names.append(threading.current_thread().name)
            return "done"

        # Test with custom name
        runner = TaskRunner(name="custom_runner")
        runner.add("task1", capture_thread_name)
        result = runner.run()

        assert result.is_ok
        assert any("custom_runner_task_runner" in name for name in executed_thread_names)

        # Test with default name
        executed_thread_names.clear()
        runner2 = TaskRunner()
        runner2.add("task2", capture_thread_name)
        result2 = runner2.run()

        assert result2.is_ok
        assert any("task_runner" in name for name in executed_thread_names)

    def test_logging_suppression(self) -> None:
        """Test that exception logging can be suppressed."""

        def failing_task() -> str:
            raise ValueError("Expected error")

        # Test with logging (default)
        runner1 = TaskRunner(suppress_logging=False)
        runner1.add("fail", failing_task)
        result1 = runner1.run()

        assert not result1.is_ok
        assert "fail" in result1.exceptions
        assert isinstance(result1.exceptions["fail"], ValueError)

        # Test with suppressed logging
        runner2 = TaskRunner(suppress_logging=True)
        runner2.add("fail", failing_task)
        result2 = runner2.run()

        assert not result2.is_ok
        assert "fail" in result2.exceptions
        assert isinstance(result2.exceptions["fail"], ValueError)

    def test_empty_results_are_preserved(self) -> None:
        """Test that empty string and zero results are preserved."""

        def return_empty_string() -> str:
            return ""

        def return_zero() -> int:
            return 0

        def return_false() -> bool:
            return False

        runner = TaskRunner()
        runner.add("empty", return_empty_string)
        runner.add("zero", return_zero)
        runner.add("false", return_false)

        result = runner.run()

        assert result.is_ok
        assert result.results["empty"] == ""
        assert result.results["zero"] == 0
        assert result.results["false"] is False

    def test_task_with_complex_arguments(self) -> None:
        """Test task execution with complex argument combinations."""

        def complex_task(
            pos_arg: str,
            *args: int,
            kw_arg: str = "default",
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "pos_arg": pos_arg,
                "args": args,
                "kw_arg": kw_arg,
                "kwargs": kwargs,
            }

        runner = TaskRunner()
        runner.add(
            "complex",
            complex_task,
            ("hello", 1, 2, 3),
            {"kw_arg": "custom", "extra": "value"},
        )

        result = runner.run()

        assert result.is_ok
        expected = {
            "pos_arg": "hello",
            "args": (1, 2, 3),
            "kw_arg": "custom",
            "kwargs": {"extra": "value"},
        }
        assert result.results["complex"] == expected

    def test_large_number_of_tasks(self) -> None:
        """Test execution with a large number of tasks."""

        def simple_task(value: int) -> int:
            return value * 2

        num_tasks = 50
        runner = TaskRunner(max_concurrent_tasks=10)

        for i in range(num_tasks):
            runner.add(f"task_{i}", simple_task, (i,))

        result = runner.run()

        assert result.is_ok
        assert len(result.results) == num_tasks
        assert len(result.exceptions) == 0

        for i in range(num_tasks):
            assert result.results[f"task_{i}"] == i * 2

    def test_exception_logging_context(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that exception logging includes task context."""

        def failing_task() -> str:
            raise ValueError("Task error")

        runner = TaskRunner(suppress_logging=False)
        runner.add("failing_task", failing_task)

        with caplog.at_level(logging.ERROR):
            result = runner.run()

        assert not result.is_ok
        # Check that the task key is included in log context
        assert any(
            record.task_key == "failing_task"  # type: ignore[attr-defined]
            for record in caplog.records
            if hasattr(record, "task_key")
        )
