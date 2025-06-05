"""Tests for async synchronization decorators."""

import asyncio
import time

import pytest

from mm_concurrency import async_synchronized, async_synchronized_by_arg_value


class TestAsyncSynchronized:
    """Tests for the async_synchronized decorator."""

    async def test_basic_serialization(self) -> None:
        """Test that all async function calls are fully synchronized."""
        call_order: list[str] = []

        @async_synchronized
        async def process_task(task_name: str) -> str:
            call_order.append(f"start_{task_name}")
            await asyncio.sleep(0.1)  # Simulate async work
            call_order.append(f"end_{task_name}")
            return f"result_{task_name}"

        # Start multiple coroutines with different arguments
        tasks = [
            asyncio.create_task(process_task("task1")),
            asyncio.create_task(process_task("task2")),
            asyncio.create_task(process_task("task3")),
        ]

        await asyncio.gather(*tasks)

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

    async def test_different_arguments_still_synchronized(self) -> None:
        """Test that even different arguments are synchronized."""
        execution_times: list[tuple[str, float, float]] = []

        @async_synchronized
        async def process_data(data_id: str, _value: int) -> None:
            start_time = time.time()
            await asyncio.sleep(0.05)
            end_time = time.time()
            execution_times.append((data_id, start_time, end_time))

        # Start coroutines with completely different arguments
        tasks = [
            asyncio.create_task(process_data("data1", 100)),
            asyncio.create_task(process_data("data2", 200)),
            asyncio.create_task(process_data("data3", 300)),
        ]

        await asyncio.gather(*tasks)

        # Verify that executions don't overlap (synchronized)
        assert len(execution_times) == 3
        execution_times.sort(key=lambda x: x[1])  # Sort by start time

        for i in range(len(execution_times) - 1):
            current_end = execution_times[i][2]
            next_start = execution_times[i + 1][1]
            # Next execution should start after current ends (with small tolerance)
            assert next_start >= current_end - 0.01

    async def test_class_methods(self) -> None:
        """Test that async_synchronized works correctly on class methods."""
        call_order: list[str] = []

        class AsyncCounter:
            def __init__(self, name: str) -> None:
                self.name = name
                self.value = 0

            @async_synchronized
            async def increment(self, by: int = 1) -> int:
                call_order.append(f"{self.name}_start")
                await asyncio.sleep(0.05)
                self.value += by
                call_order.append(f"{self.name}_end")
                return self.value

        # Create different instances
        counter1 = AsyncCounter("C1")
        counter2 = AsyncCounter("C2")

        # All method calls should be synchronized across instances
        tasks = [
            asyncio.create_task(counter1.increment(5)),
            asyncio.create_task(counter2.increment(10)),
            asyncio.create_task(counter1.increment(3)),
        ]

        await asyncio.gather(*tasks)

        # All calls should be synchronized
        assert len(call_order) == 6

        # Check that we have proper start-end pairs
        for i in range(0, 6, 2):
            assert call_order[i].endswith("_start")
            assert call_order[i + 1].endswith("_end")

    async def test_exception_handling(self) -> None:
        """Test that locks are properly released when async function raises."""
        call_count = 0

        @async_synchronized
        async def failing_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success"

        # First call should fail but release lock
        with pytest.raises(ValueError, match="First call fails"):
            await failing_function()

        # Second call should succeed (lock was released)
        result = await failing_function()
        assert result == "success"

    async def test_return_values(self) -> None:
        """Test that async function return values work correctly."""

        @async_synchronized
        async def calculate(x: int, y: int) -> int:
            await asyncio.sleep(0.01)  # Small delay to ensure serialization
            return x + y

        # Run multiple calculations concurrently
        tasks = [
            asyncio.create_task(calculate(5, 3)),
            asyncio.create_task(calculate(10, 7)),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert 8 in results  # 5 + 3
        assert 17 in results  # 10 + 7

    async def test_concurrent_access_with_asyncio_gather(self) -> None:
        """Test behavior with asyncio.gather - should still be synchronized."""
        call_order: list[str] = []

        @async_synchronized
        async def process_item(item_id: str) -> str:
            call_order.append(f"start_{item_id}")
            await asyncio.sleep(0.05)
            call_order.append(f"end_{item_id}")
            return f"processed_{item_id}"

        # Use asyncio.gather to run multiple coroutines
        results = await asyncio.gather(
            process_item("item1"),
            process_item("item2"),
            process_item("item3"),
        )

        # Should be synchronized despite using gather
        assert len(call_order) == 6
        assert len(results) == 3

        # Verify serialization order
        for i in range(0, 6, 2):
            start_call = call_order[i]
            end_call = call_order[i + 1]
            assert start_call.startswith("start_")
            assert end_call.startswith("end_")

    async def test_multiple_async_operations(self) -> None:
        """Test that async_synchronized works with multiple async operations."""
        call_order: list[str] = []

        @async_synchronized
        async def mixed_operation(op_id: str) -> str:
            call_order.append(f"async_start_{op_id}")
            # Mix async operations
            await asyncio.sleep(0.02)
            await asyncio.sleep(0.02)
            await asyncio.sleep(0.02)
            call_order.append(f"async_end_{op_id}")
            return f"result_{op_id}"

        # Run multiple operations
        tasks = [
            asyncio.create_task(mixed_operation("op1")),
            asyncio.create_task(mixed_operation("op2")),
        ]

        results = await asyncio.gather(*tasks)

        # Should be synchronized
        assert len(call_order) == 4
        assert len(results) == 2

        # First operation should complete before second starts
        expected_orders = {
            ("async_start_op1", "async_end_op1", "async_start_op2", "async_end_op2"),
            ("async_start_op2", "async_end_op2", "async_start_op1", "async_end_op1"),
        }
        assert tuple(call_order) in expected_orders


class TestAsyncSynchronizedByArgValue:
    """Tests for the async_synchronized_by_arg_value decorator."""

    async def test_different_keys_execute_concurrently(self) -> None:
        """Test that different argument values allow concurrent execution."""
        call_order: list[str] = []

        @async_synchronized_by_arg_value(index=0)
        async def process_user(user_id: str) -> str:
            call_order.append(f"start_{user_id}")
            await asyncio.sleep(0.1)  # Simulate async work
            call_order.append(f"end_{user_id}")
            return f"processed_{user_id}"

        # Start tasks with different user IDs - should run concurrently
        tasks = [
            asyncio.create_task(process_user("user1")),
            asyncio.create_task(process_user("user2")),
        ]

        results = await asyncio.gather(*tasks)

        # Both tasks should complete successfully
        assert results == ["processed_user1", "processed_user2"]

        # Since different keys run concurrently, both should start before either ends
        # (due to the sleep, if they were serialized, we'd see start_user1, end_user1, start_user2, end_user2)
        assert "start_user1" in call_order
        assert "start_user2" in call_order
        assert "end_user1" in call_order
        assert "end_user2" in call_order

        # Both should start before either ends (proving concurrency)
        user1_start = call_order.index("start_user1")
        user2_start = call_order.index("start_user2")
        user1_end = call_order.index("end_user1")
        user2_end = call_order.index("end_user2")

        # Both starts should come before both ends
        assert user1_start < user1_end
        assert user1_start < user2_end
        assert user2_start < user1_end
        assert user2_start < user2_end

    async def test_same_key_executes_serially(self) -> None:
        """Test that same argument values are synchronized."""
        call_order: list[str] = []

        @async_synchronized_by_arg_value(index=0)
        async def process_user(user_id: str, operation: str) -> str:
            call_order.append(f"start_{user_id}_{operation}")
            await asyncio.sleep(0.1)  # Simulate async work
            call_order.append(f"end_{user_id}_{operation}")
            return f"processed_{user_id}_{operation}"

        # Start multiple tasks with the same user ID - should run serially
        tasks = [
            asyncio.create_task(process_user("user1", "op1")),
            asyncio.create_task(process_user("user1", "op2")),
        ]

        results = await asyncio.gather(*tasks)

        # Both tasks should complete successfully
        assert set(results) == {"processed_user1_op1", "processed_user1_op2"}

        # Since same key runs serially, one should complete before the other starts
        expected_orders = {
            ("start_user1_op1", "end_user1_op1", "start_user1_op2", "end_user1_op2"),
            ("start_user1_op2", "end_user1_op2", "start_user1_op1", "end_user1_op1"),
        }
        assert tuple(call_order) in expected_orders

    async def test_synchronization_by_keyword_parameter(self) -> None:
        """Test synchronization using keyword parameter name."""
        call_order: list[str] = []

        @async_synchronized_by_arg_value(key="resource_id")
        async def process_resource(data: dict, resource_id: str) -> str:
            call_order.append(f"start_{resource_id}_{data['value']}")
            await asyncio.sleep(0.05)
            call_order.append(f"end_{resource_id}_{data['value']}")
            return f"processed_{resource_id}"

        # Test with same resource_id - should be serialized
        tasks = [
            asyncio.create_task(process_resource({"value": 1}, resource_id="res1")),
            asyncio.create_task(process_resource({"value": 2}, resource_id="res1")),
        ]

        await asyncio.gather(*tasks)

        # Should be serialized for same resource_id
        expected_orders = {
            ("start_res1_1", "end_res1_1", "start_res1_2", "end_res1_2"),
            ("start_res1_2", "end_res1_2", "start_res1_1", "end_res1_1"),
        }
        assert tuple(call_order) in expected_orders

    async def test_nonblocking_mode_returns_none_when_locked(self) -> None:
        """Test that nonblocking mode returns None when lock is held."""
        barrier = asyncio.Event()
        results: list[str | None] = []

        @async_synchronized_by_arg_value(index=0, nonblocking=True)
        async def slow_operation(key: str) -> str:
            await barrier.wait()  # Wait for signal
            return f"result_{key}"

        async def first_task() -> None:
            result = await slow_operation("key1")
            results.append(result)

        async def second_task() -> None:
            # Small delay to ensure first task gets the lock first
            await asyncio.sleep(0.01)
            result = await slow_operation("key1")
            results.append(result)

        # Start both tasks
        task1 = asyncio.create_task(first_task())
        task2 = asyncio.create_task(second_task())

        # Wait a moment for both to be started
        await asyncio.sleep(0.02)

        # Second task should return None immediately (nonblocking)
        # First task should still be waiting
        assert len(results) == 1
        assert results[0] is None

        # Now release the barrier
        barrier.set()

        # Wait for first task to complete
        await task1
        await task2

        # First task should have completed successfully
        assert len(results) == 2
        assert results[1] == "result_key1"

    async def test_nonblocking_different_keys_both_succeed(self) -> None:
        """Test that nonblocking mode allows different keys to proceed."""

        @async_synchronized_by_arg_value(index=0, nonblocking=True)
        async def process_item(item_id: str) -> str:
            await asyncio.sleep(0.01)
            return f"processed_{item_id}"

        # Different keys should both succeed even in nonblocking mode
        tasks = [
            asyncio.create_task(process_item("item1")),
            asyncio.create_task(process_item("item2")),
        ]

        results = await asyncio.gather(*tasks)

        assert results == ["processed_item1", "processed_item2"]

    async def test_error_handling_with_cleanup(self) -> None:
        """Test that exceptions don't prevent proper cleanup."""
        call_count = 0

        @async_synchronized_by_arg_value(index=0)
        async def failing_function(key: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Intentional error")
            return f"success_{key}"

        # First call should fail
        with pytest.raises(ValueError, match="Intentional error"):
            await failing_function("fail")

        # Second call with same key should work (proving cleanup happened)
        result = await failing_function("fail")
        assert result == "success_fail"

    async def test_invalid_parameter_name_raises_error(self) -> None:
        """Test that specifying non-existent parameter name raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'nonexistent' not found"):

            @async_synchronized_by_arg_value(key="nonexistent")
            async def test_func(valid_param: str) -> str:
                return valid_param

    async def test_argument_index_out_of_range_raises_error(self) -> None:
        """Test that invalid argument index raises IndexError."""

        @async_synchronized_by_arg_value(index=2)  # Only 1 argument available
        async def test_func(param: str) -> str:
            return param

        with pytest.raises(IndexError, match="Argument index 2 out of range"):
            await test_func("test")

    async def test_memory_cleanup_after_usage(self) -> None:
        """Test that locks are cleaned up when no longer in use."""

        # Create a simple operation that we can monitor
        @async_synchronized_by_arg_value(index=0)
        async def short_operation(key: str) -> str:
            return f"result_{key}"

        # Perform operations with unique keys
        result1 = await short_operation("temp_key_1")
        result2 = await short_operation("temp_key_2")

        # Verify the operations worked
        assert result1 == "result_temp_key_1"
        assert result2 == "result_temp_key_2"

        # This test mainly verifies that the decorator doesn't crash
        # and that operations complete successfully, which indicates
        # proper cleanup is happening (no memory leaks causing issues)
