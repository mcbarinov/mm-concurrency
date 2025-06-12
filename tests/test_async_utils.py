"""Tests for async_utils module."""

import asyncio
import time

import pytest

from mm_concurrency.async_utils import async_timeout


class TestAsyncTimeout:
    """Test cases for async_timeout context manager."""

    async def test_successful_operation_within_timeout(self) -> None:
        """Test that operations completing within timeout work normally."""

        async def quick_operation() -> str:
            await asyncio.sleep(0.1)
            return "success"

        async with async_timeout(1.0):
            result = await quick_operation()

        assert result == "success"

    async def test_operation_exceeds_timeout(self) -> None:
        """Test that operations exceeding timeout raise TimeoutError."""

        async def slow_operation() -> str:
            await asyncio.sleep(1.0)
            return "too slow"

        with pytest.raises(TimeoutError):
            async with async_timeout(0.1):
                await slow_operation()

    async def test_timeout_precision(self) -> None:
        """Test that timeout is reasonably precise."""
        start_time = time.time()

        with pytest.raises(TimeoutError):
            async with async_timeout(0.2):
                await asyncio.sleep(1.0)

        elapsed = time.time() - start_time
        # Should timeout after ~0.2 seconds, allow some margin
        assert 0.15 < elapsed < 0.35

    async def test_nested_timeouts(self) -> None:
        """Test that nested timeouts work correctly."""

        async def operation() -> str:
            await asyncio.sleep(0.3)
            return "done"

        # Inner timeout should trigger first
        with pytest.raises(TimeoutError):
            async with async_timeout(1.0):  # Outer timeout
                async with async_timeout(0.1):  # Inner timeout (shorter)
                    await operation()

    async def test_timeout_with_task_group(self) -> None:
        """Test timeout combined with TaskGroup."""

        async def task(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"task_{delay}"

        # Should timeout due to slow task
        with pytest.raises(TimeoutError):
            async with async_timeout(0.2):
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(task(0.1))
                    tg.create_task(task(0.5))  # This will cause timeout

    async def test_immediate_timeout(self) -> None:
        """Test very short timeout."""
        with pytest.raises(TimeoutError):
            async with async_timeout(0.001):
                await asyncio.sleep(0.1)

    async def test_zero_timeout_validation(self) -> None:
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            async with async_timeout(0):
                pass

    async def test_negative_timeout_validation(self) -> None:
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            async with async_timeout(-1.0):
                pass

    async def test_multiple_operations_in_timeout(self) -> None:
        """Test multiple sequential operations within timeout."""
        results = []

        async with async_timeout(0.5):
            await asyncio.sleep(0.1)
            results.append("first")
            await asyncio.sleep(0.1)
            results.append("second")
            await asyncio.sleep(0.1)
            results.append("third")

        assert results == ["first", "second", "third"]

    async def test_timeout_exception_chaining(self) -> None:
        """Test that TimeoutError doesn't chain the original asyncio.TimeoutError."""
        try:
            async with async_timeout(0.1):
                await asyncio.sleep(1.0)
        except TimeoutError as e:
            # Should not chain the original asyncio.TimeoutError
            assert e.__cause__ is None

    async def test_context_manager_cleanup(self) -> None:
        """Test that context manager cleans up properly even with exceptions."""

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            async with async_timeout(1.0):
                raise CustomError("test error")

        # Should be able to use timeout again after exception
        async with async_timeout(1.0):
            await asyncio.sleep(0.1)

    async def test_concurrent_timeouts(self) -> None:
        """Test multiple concurrent timeout contexts."""

        async def timed_task(timeout: float, sleep_time: float) -> str:
            try:
                async with async_timeout(timeout):
                    await asyncio.sleep(sleep_time)
            except TimeoutError:
                return "timeout"
            else:
                return "success"

        # Run multiple timeout contexts concurrently
        results = await asyncio.gather(
            timed_task(0.2, 0.1),  # Should succeed
            timed_task(0.1, 0.2),  # Should timeout
            timed_task(0.3, 0.15),  # Should succeed
        )

        assert results == ["success", "timeout", "success"]
