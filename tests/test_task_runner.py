import time
from typing import Any

from mm_concurrency import TaskRunner


class TestTaskRunner:
    """Tests for the TaskRunner class."""

    def test_successful_execution(self) -> None:
        """Test basic successful task execution and result collection."""

        def add_numbers(x: int, y: int) -> int:
            return x + y

        def multiply(x: int, y: int) -> int:
            return x * y

        runner = TaskRunner(max_workers=2)
        runner.add("add", add_numbers, (5, 3))
        runner.add("multiply", multiply, (4, 7))
        runner.run()

        # Check results
        assert not runner.has_errors
        assert not runner.has_timeout
        assert len(runner.results) == 2
        assert runner.results["add"] == 8
        assert runner.results["multiply"] == 28
        assert len(runner.exceptions) == 0

    def test_task_with_kwargs(self) -> None:
        """Test tasks with keyword arguments."""

        def format_message(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        runner = TaskRunner()
        runner.add("task1", format_message, ("Alice",))
        runner.add("task2", format_message, ("Bob",), {"greeting": "Hi"})
        runner.run()

        assert not runner.has_errors
        assert runner.results["task1"] == "Hello, Alice!"
        assert runner.results["task2"] == "Hi, Bob!"

    def test_exception_handling(self) -> None:
        """Test that exceptions are properly captured and don't affect other tasks."""

        def success_task() -> str:
            return "success"

        def failing_task() -> None:
            raise ValueError("Task failed")

        def another_success() -> int:
            return 42

        runner = TaskRunner(max_workers=3)
        runner.add("success1", success_task)
        runner.add("failure", failing_task)
        runner.add("success2", another_success)
        runner.run()

        # Check that failure doesn't prevent other tasks
        assert runner.has_errors
        assert not runner.has_timeout
        assert len(runner.results) == 2  # Two successful tasks
        assert len(runner.exceptions) == 1  # One failed task

        assert runner.results["success1"] == "success"
        assert runner.results["success2"] == 42
        assert isinstance(runner.exceptions["failure"], ValueError)
        assert str(runner.exceptions["failure"]) == "Task failed"

    def test_timeout_handling(self) -> None:
        """Test timeout behavior with slow tasks."""

        def slow_task(delay: float) -> str:
            time.sleep(delay)
            return "completed"

        def fast_task() -> str:
            return "fast"

        runner = TaskRunner(max_workers=2, timeout=0.1)  # 100ms timeout
        runner.add("slow", slow_task, (0.2,))  # 200ms task
        runner.add("fast", fast_task)
        runner.run()

        # Should timeout
        assert runner.has_errors
        assert runner.has_timeout
        # Results may vary depending on timing, but at least we test the flags

    def test_concurrent_execution(self) -> None:
        """Test that tasks actually run concurrently."""
        execution_order: list[str] = []

        def task_with_delay(name: str, delay: float) -> str:
            execution_order.append(f"{name}_start")
            time.sleep(delay)
            execution_order.append(f"{name}_end")
            return f"{name}_result"

        runner = TaskRunner(max_workers=3)
        runner.add("task1", task_with_delay, ("A", 0.1))
        runner.add("task2", task_with_delay, ("B", 0.1))
        runner.add("task3", task_with_delay, ("C", 0.1))

        start_time = time.time()
        runner.run()
        end_time = time.time()

        # Should complete in roughly 0.1 seconds (concurrent), not 0.3 seconds (sequential)
        assert end_time - start_time < 0.25  # Allow some overhead

        # All tasks should complete successfully
        assert not runner.has_errors
        assert len(runner.results) == 3

    def test_no_tasks(self) -> None:
        """Test behavior when no tasks are added."""
        runner = TaskRunner()
        runner.run()

        assert not runner.has_errors
        assert not runner.has_timeout
        assert len(runner.results) == 0
        assert len(runner.exceptions) == 0

    def test_duplicate_keys(self) -> None:
        """Test behavior with duplicate task keys."""

        def simple_task(value: int) -> int:
            return value * 2

        runner = TaskRunner()
        runner.add("task", simple_task, (5,))
        runner.add("task", simple_task, (10,))  # Same key
        runner.run()

        # Should have 2 tasks but only 1 result (last one wins)
        assert not runner.has_errors
        assert len(runner.tasks) == 2
        assert len(runner.results) == 1
        # Result should be from one of the tasks (could be either due to concurrency)
        assert runner.results["task"] in [10, 20]

    def test_thread_prefix_configuration(self) -> None:
        """Test that thread prefix is configurable."""

        def simple_task() -> str:
            import threading

            return threading.current_thread().name

        runner = TaskRunner(thread_prefix="test_threads")
        runner.add("task", simple_task)
        runner.run()

        assert not runner.has_errors
        thread_name = runner.results["task"]
        assert isinstance(thread_name, str)
        assert "test_threads" in thread_name

    def test_mixed_success_and_failure(self) -> None:
        """Test comprehensive scenario with mix of successful and failed tasks."""

        def divide(x: int, y: int) -> float:
            return x / y

        runner = TaskRunner(max_workers=4)
        runner.add("div1", divide, (10, 2))  # Success: 5.0
        runner.add("div2", divide, (15, 3))  # Success: 5.0
        runner.add("div3", divide, (10, 0))  # Failure: ZeroDivisionError
        runner.add("div4", divide, (20, 4))  # Success: 5.0
        runner.run()

        assert runner.has_errors
        assert not runner.has_timeout
        assert len(runner.results) == 3  # Three successful
        assert len(runner.exceptions) == 1  # One failed

        # Check successful results
        assert runner.results["div1"] == 5.0
        assert runner.results["div2"] == 5.0
        assert runner.results["div4"] == 5.0

        # Check exception
        assert isinstance(runner.exceptions["div3"], ZeroDivisionError)

    def test_task_return_types(self) -> None:
        """Test that various return types are handled correctly."""

        def return_none() -> None:
            return None

        def return_dict() -> dict[str, Any]:
            return {"key": "value"}

        def return_list() -> list[int]:
            return [1, 2, 3]

        runner = TaskRunner()
        runner.add("none", return_none)
        runner.add("dict", return_dict)
        runner.add("list", return_list)
        runner.run()

        assert not runner.has_errors
        assert runner.results["none"] is None
        assert runner.results["dict"] == {"key": "value"}
        assert runner.results["list"] == [1, 2, 3]
