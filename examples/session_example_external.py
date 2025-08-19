"""
Session example where the client is responsible for managing the session.

Check out session_example_internal.py for an example where the session is managed by the
orchestrator.
"""

import asyncio
import os
import logging
from openai.types.responses import EasyInputMessageParam
from src.session import Session
from openai import AsyncOpenAI
from openai.types import Reasoning

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    session_id = "conversation_1"
    session = Session(session_id)

    # First Turn
    input_text = "What city is the Golden Gate Bridge in?"

    session.add_items(
        [EasyInputMessageParam(content=input_text, role="user", type="message")]
    )
    logger.info("First Turn")
    logger.info("User: %s", input_text)

    response = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
        model="gpt-5-nano",
        input=session.get_items(),
        reasoning=Reasoning(summary="auto"),
    )

    for output in response.output:
        logger.info("Assistant: %s", output)

    session.add_items(response.output)

    # Second Turn
    input_text = "What state is it in?"

    logger.info("Second Turn")
    logger.info("User: %s", input_text)

    session.add_items(
        [EasyInputMessageParam(content=input_text, role="user", type="message")]
    )

    response = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
        model="gpt-5-nano",
        input=session.get_items(),
    )

    for output in response.output:
        logger.info("Assistant: %s", output)

    session.add_items(response.output)

    # Third Turn
    input_text = "What country is it in?"

    logger.info("Third Turn")
    logger.info("User: %s", input_text)

    session.add_items(
        [EasyInputMessageParam(content=input_text, role="user", type="message")]
    )

    response = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
        model="gpt-5-nano",
        input=session.get_items(),
    )

    for output in response.output:
        logger.info("Assistant: %s", output)

    session.add_items(response.output)

    logger.info("================================================")
    logger.info("Conversation Complete")
    logger.info("================================================")

    logger.info("Notice how the agent remembered the context from previous turns!")
    logger.info("Sessions automatically handles conversation history.")

    # Demonstrate the limit parameter - get only the latest 2 items
    logger.info("\n=== Latest Items Demo ===")
    latest_items = session.get_items(limit=2)
    logger.info("Latest 2 items:")
    for i, msg in enumerate(latest_items, 1):
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")
        logger.info("  %s. %s: %s", i, role, content)

    logger.info("\nFetched %s out of total conversation history.", len(latest_items))

    # Get all items to show the difference
    all_items = session.get_items()
    logger.info("Total items in session: %s", len(all_items))


if __name__ == "__main__":
    asyncio.run(main())
