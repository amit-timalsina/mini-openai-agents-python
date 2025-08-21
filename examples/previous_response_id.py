"""This demonstrates usage of the `previous_response_id` parameter to continue a conversation.
The second run passes the previous response ID to the model, which allows it to continue the
conversation without re-sending the previous messages.

Notes:
Responses are only stored for 30 days as of this writing, so in production you should
store the response ID along with an expiration date; if the response is no longer valid,
you'll need to re-send the previous conversation history.
"""

import asyncio
import logging
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_param import Message

from src.orchestrator import run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Example of using the `previous_response_id` parameter to continue a conversation."""

    prepared_input: ResponseInputParam = [
        Message(
            role="user",
            content=[
                ResponseInputTextParam(
                    type="input_text", text="What is the country with Mount Everest?"
                )
            ],
        )
    ]

    response, _ = await run(prepared_input)
    logger.info(f"First response: {response}")

    previous_response_id = response.id
    prepared_input = [
        Message(
            role="user",
            content=[
                ResponseInputTextParam(
                    type="input_text", text="What is the capital of that country?"
                )
            ],
        )
    ]
    response, _ = await run(prepared_input, previous_response_id)
    logger.info(f"Second response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
