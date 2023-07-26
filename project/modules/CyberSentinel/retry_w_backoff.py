import asyncio
import logging
from typing import Callable, Awaitable

async def retry_with_backoff(
    func: Callable[..., Awaitable],
    max_retries: int = 3,
    backoff_values: list[float] = [4],
) -> None:
    """
    Retry the given async function with exponential backoff.

    Args:
        func (Callable[..., Awaitable]): The async function to retry.
        max_retries (int): Maximum number of retries (default: 3).
        backoff_values (list[float]): List of backoff values in seconds (default: [4]).

    Returns:
        None
    """
    for retry in range(max_retries):
        try:
            await func()
            break  # Success, no need to retry
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            if retry == max_retries - 1:
                logging.error(f"Max retries exceeded.")
                break  # Max retries reached, give up
            delay = backoff_values[min(retry, len(backoff_values) - 1)]
            logging.warning(f"Retrying in {delay} seconds... ({retry + 1}/{max_retries})")
            await asyncio.sleep(delay)
