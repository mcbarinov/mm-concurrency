"""Tests for async synchronization decorators."""

import asyncio
import time

import pytest

from mm_concurrency import async_synchronized


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
