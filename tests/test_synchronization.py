import threading
import time
from typing import Any

import pytest

from mm_concurrency import synchronized, synchronized_by_arg_value


class TestSynchronized:
    """Tests for the synchronized decorator."""

    def test_basic_serialization(self) -> None:
        """Test that all function calls are fully synchronized."""
        call_order: list[str] = []

        @synchronized
        def process_task(task_name: str) -> str:
            call_order.append(f"start_{task_name}")
            time.sleep(0.1)  # Simulate work
            call_order.append(f"end_{task_name}")
            return f"result_{task_name}"

        # Start multiple threads with different arguments
        def worker1() -> None:
            process_task("task1")

        def worker2() -> None:
            process_task("task2")

        def worker3() -> None:
            process_task("task3")

        threads = [threading.Thread(target=worker1), threading.Thread(target=worker2), threading.Thread(target=worker3)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All calls should be fully synchronized - complete task before starting next
        assert len(call_order) == 6

        # Check that each task completes before the next starts
        for i in range(0, 6, 2):
            start_call = call_order[i]
            end_call = call_order[i + 1]
            assert start_call.startswith("start_")
            assert end_call.startswith("end_")
            # Extract task name and verify they match
            task_from_start = start_call.split("_", 1)[1]
            task_from_end = end_call.split("_", 1)[1]
            assert task_from_start == task_from_end

    def test_different_arguments_still_synchronized(self) -> None:
        """Test that even different arguments are synchronized (unlike synchronized_by_arg_value)."""
        execution_times: list[tuple[str, float, float]] = []

        @synchronized
        def process_data(data_id: str, _value: int) -> None:
            start_time = time.time()
            time.sleep(0.05)
            end_time = time.time()
            execution_times.append((data_id, start_time, end_time))

        # Start threads with completely different arguments
        threads = [
            threading.Thread(target=process_data, args=("data1", 100)),
            threading.Thread(target=process_data, args=("data2", 200)),
            threading.Thread(target=process_data, args=("data3", 300)),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify that executions don't overlap (synchronized)
        assert len(execution_times) == 3
        execution_times.sort(key=lambda x: x[1])  # Sort by start time

        for i in range(len(execution_times) - 1):
            current_end = execution_times[i][2]
            next_start = execution_times[i + 1][1]
            # Next execution should start after current ends (with small tolerance)
            assert next_start >= current_end - 0.01

    def test_class_methods(self) -> None:
        """Test that synchronized works correctly on class methods."""
        call_order: list[str] = []

        class Counter:
            def __init__(self, name: str) -> None:
                self.name = name
                self.value = 0

            @synchronized
            def increment(self, by: int = 1) -> int:
                call_order.append(f"{self.name}_start")
                time.sleep(0.05)
                self.value += by
                call_order.append(f"{self.name}_end")
                return self.value

        # Create different instances
        counter1 = Counter("C1")
        counter2 = Counter("C2")

        # All method calls should be synchronized across instances
        def worker1() -> None:
            counter1.increment(5)

        def worker2() -> None:
            counter2.increment(10)

        def worker3() -> None:
            counter1.increment(3)

        threads = [threading.Thread(target=worker1), threading.Thread(target=worker2), threading.Thread(target=worker3)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All calls should be synchronized
        assert len(call_order) == 6

        # Check that we have proper start-end pairs
        for i in range(0, 6, 2):
            assert call_order[i].endswith("_start")
            assert call_order[i + 1].endswith("_end")

    def test_exception_handling(self) -> None:
        """Test that locks are properly released when function raises."""
        call_count = 0

        @synchronized
        def failing_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success"

        # First call should fail but release lock
        with pytest.raises(ValueError, match="First call fails"):
            failing_function()

        # Second call should succeed (lock was released)
        result = failing_function()
        assert result == "success"

    def test_return_values(self) -> None:
        """Test that function return values work correctly."""

        @synchronized
        def calculate(x: int, y: int) -> int:
            time.sleep(0.01)  # Small delay to ensure serialization
            return x + y

        results: list[int] = []

        def worker1() -> None:
            result = calculate(5, 3)
            results.append(result)

        def worker2() -> None:
            result = calculate(10, 7)
            results.append(result)

        threads = [threading.Thread(target=worker1), threading.Thread(target=worker2)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 2
        assert 8 in results  # 5 + 3
        assert 17 in results  # 10 + 7


class TestSynchronizedByArgValue:
    """Tests for the synchronized_by_arg_value decorator."""

    def test_basic_locking_by_index(self) -> None:
        """Test that function calls with same argument value are synchronized."""
        call_order: list[str] = []

        @synchronized_by_arg_value(index=0)
        def slow_process(key: str) -> str:
            call_order.append(f"start_{key}")
            time.sleep(0.1)  # Simulate work
            call_order.append(f"end_{key}")
            return f"result_{key}"

        # Start two threads with same key - should be synchronized
        def worker1() -> None:
            slow_process("same_key")

        def worker2() -> None:
            slow_process("same_key")

        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Calls should be fully synchronized - one complete before other starts
        assert len(call_order) == 4
        assert call_order[0] == "start_same_key"
        assert call_order[1] == "end_same_key"
        assert call_order[2] == "start_same_key"
        assert call_order[3] == "end_same_key"

    def test_parallel_execution_different_keys(self) -> None:
        """Test that different argument values can execute in parallel."""
        start_times: dict[str, float] = {}
        end_times: dict[str, float] = {}

        @synchronized_by_arg_value(index=0)
        def slow_process(key: str) -> None:
            start_times[key] = time.time()
            time.sleep(0.1)
            end_times[key] = time.time()

        # Start threads with different keys - should run in parallel
        threads = []
        for key in ["key1", "key2"]:
            thread = threading.Thread(target=slow_process, args=(key,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Both should have started around the same time (parallel execution)
        time_diff = abs(start_times["key1"] - start_times["key2"])
        assert time_diff < 0.05  # Started within 50ms of each other

    def test_locking_by_parameter_name(self) -> None:
        """Test locking by parameter name instead of index."""
        results: list[str] = []

        @synchronized_by_arg_value(key="user_id")
        def process_user(user_id: str, _data: dict[str, Any]) -> None:
            results.append(f"processing_{user_id}")
            time.sleep(0.05)
            results.append(f"done_{user_id}")

        # Test both positional and keyword argument styles
        def worker1() -> None:
            process_user("user123", {"type": "update"})

        def worker2() -> None:
            process_user(user_id="user123", _data={"type": "delete"})

        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Should be synchronized since same user_id
        assert results == ["processing_user123", "done_user123", "processing_user123", "done_user123"]

    def test_nonblocking_mode(self) -> None:
        """Test nonblocking mode returns None when lock is held."""
        results: list[str | None] = []

        @synchronized_by_arg_value(nonblocking=True)
        def try_process(key: str) -> str:
            time.sleep(0.1)
            return f"processed_{key}"

        def worker1() -> None:
            result = try_process("shared_key")
            results.append(result)

        def worker2() -> None:
            # Start slightly after worker1 to ensure lock is held
            time.sleep(0.02)
            result = try_process("shared_key")
            results.append(result)

        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # One should succeed, one should get None
        assert len(results) == 2
        assert "processed_shared_key" in results
        assert None in results

    def test_error_cases(self) -> None:
        """Test various error conditions."""
        # Invalid parameter name
        with pytest.raises(ValueError, match="Parameter 'nonexistent' not found"):

            @synchronized_by_arg_value(key="nonexistent")
            def func(param: str) -> None:
                pass

        # Index out of range
        @synchronized_by_arg_value(index=2)  # Looking for 3rd argument (index 2)
        def needs_two_args(first: str, second: str) -> None:
            pass

        with pytest.raises(IndexError, match="Argument index 2 out of range"):
            needs_two_args("only_one_arg", "second_arg")

    def test_exception_handling(self) -> None:
        """Test that locks are properly released even when function raises."""
        call_count = 0

        @synchronized_by_arg_value()
        def failing_function(_key: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")

        # First call should fail but release lock
        with pytest.raises(ValueError):
            failing_function("test_key")

        # Second call should succeed (lock was released)
        failing_function("test_key")  # Should not hang

    def test_mixed_argument_styles(self) -> None:
        """Test function works with both positional and keyword arguments."""

        @synchronized_by_arg_value(key="user_id")
        def update_user(user_id: str, name: str, age: int) -> str:
            return f"{user_id}_{name}_{age}"

        # Different ways to call the same function
        result1 = update_user("123", "Alice", 30)
        result2 = update_user(user_id="456", name="Bob", age=25)
        result3 = update_user("789", name="Charlie", age=35)

        assert result1 == "123_Alice_30"
        assert result2 == "456_Bob_25"
        assert result3 == "789_Charlie_35"

    def test_class_methods(self) -> None:
        """Test that synchronized_by_arg_value works correctly on class methods."""
        call_order: list[str] = []

        class UserProcessor:
            def __init__(self, name: str) -> None:
                self.name = name

            @synchronized_by_arg_value(key="user_id")
            def process_user(self, user_id: str, action: str) -> str:
                call_order.append(f"{self.name}_start_{user_id}_{action}")
                time.sleep(0.05)
                call_order.append(f"{self.name}_end_{user_id}_{action}")
                return f"{self.name}_processed_{user_id}_{action}"

        # Create two different instances
        processor1 = UserProcessor("P1")
        processor2 = UserProcessor("P2")

        # Test that same user_id is synchronized across different instances
        def worker1() -> None:
            processor1.process_user("user123", "update")

        def worker2() -> None:
            processor2.process_user("user123", "delete")  # Same user_id

        def worker3() -> None:
            processor1.process_user("user456", "create")  # Different user_id

        threads = [threading.Thread(target=worker1), threading.Thread(target=worker2), threading.Thread(target=worker3)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Calls with same user_id (user123) should be synchronized
        # Call with different user_id (user456) can run in parallel
        assert len(call_order) == 6

        # Find positions of user123 calls
        user123_calls = [i for i, call in enumerate(call_order) if "user123" in call]

        # user123 calls should be consecutive (synchronized)
        # We should see either P1_start -> P1_end -> P2_start -> P2_end
        # or P2_start -> P2_end -> P1_start -> P1_end
        user123_subsequence = [call_order[i] for i in user123_calls]

        # Check that starts and ends are properly paired
        assert len([call for call in user123_subsequence if "start" in call]) == 2
        assert len([call for call in user123_subsequence if "end" in call]) == 2

        # Verify that we have start-end pairs (serialization)
        start_indices = [i for i, call in enumerate(user123_subsequence) if "start" in call]
        end_indices = [i for i, call in enumerate(user123_subsequence) if "end" in call]

        # Should be either [0,2] and [1,3] or some valid pairing
        assert abs(start_indices[0] - end_indices[0]) == 1 or abs(start_indices[1] - end_indices[1]) == 1
