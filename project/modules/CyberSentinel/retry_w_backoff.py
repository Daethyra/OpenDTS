import asyncio
import logging
import random
from typing import Any, Callable, Awaitable, Tuple, Type

async def retry_with_backoff(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1,
    backoff_factor: float = 2,
    retry_exceptions: Tuple[Type[Exception]] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Retry the given async function with exponential backoff.

    Args:
        func (Callable[..., Awaitable[Any]]): The async function to retry.
        *args (Any): Optional positional arguments to pass to the function.
        max_retries (int): Optional. Maximum number of retries (default: 3).
        initial_delay (float): Optional. Initial delay in seconds (default: 1).
        backoff_factor (float): Optional. Factor to multiply the delay by after each retry (default: 2).
        retry_exceptions (Tuple[Type[Exception]]): Optional. Exceptions to catch for retries (default: all exceptions).
        **kwargs (Any): Optional keyword arguments to pass to the function.

    Returns:
        The result of the successful function call, or raises the last exception if all retries fail.
    """
    delay = initial_delay
    last_exception = None
    for retry in range(max_retries):
        try:
            return await func(*args, **kwargs)  # Return the result on success
        except retry_exceptions as e:
            last_exception = e
            logging.error(f"An error occurred: {e}")
            if retry == max_retries - 1:
                logging.error(f"Max retries exceeded.")
                raise last_exception  # Raise the last exception if max retries reached
            jitter = random.uniform(0.5, 1.5)  # Random factor between 0.5 and 1.5
            actual_delay = delay * jitter
            logging.warning(f"Retrying in {actual_delay} seconds... ({retry + 1}/{max_retries})")
            await asyncio.sleep(actual_delay)
            delay *= backoff_factor  # Increase the delay for the next retry
