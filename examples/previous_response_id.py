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
from openai import AsyncOpenAI
import os
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_param import Message
from openai.types.responses import (
    ResponseReasoningItem,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types import Reasoning
from openai.types.responses import Response

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def run(
    input: ResponseInputParam,
    previous_response_id: str | None = None,
    max_iterations: int = 10,
) -> Response:
    """
    Abstract the logic of calling llm in a loop for agentic behaviour.

    This acts like orchestrator.
    """

    current_iteration = 0
    agent_should_stop = False
    while True:
        if current_iteration >= max_iterations:
            logger.info("Max iterations reached. Exiting.")
            break

        response = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
            model="gpt-5-nano",
            instructions="You are a helpful assistant.",
            input=input,
            reasoning=Reasoning(summary="auto"),
            previous_response_id=previous_response_id,
        )

        previous_response_id = response.id

        for output in response.output:
            if isinstance(output, ResponseReasoningItem):
                if output.summary:
                    for summary_item in output.summary:
                        logger.info(f"Reasoning summary: {summary_item.text}")
                else:
                    logger.info("Reasoning summary: None")
            elif isinstance(output, ResponseOutputMessage):
                if isinstance(output.content[0], ResponseOutputText):
                    logger.info(f"Final response: {output.content[0].text}")
                    agent_should_stop = True
                else:
                    logger.warning(f"Unsupported output type: {output.content[0]}")
            else:
                logger.warning(f"Unsupported output type: {output}")

        breakpoint()
        if agent_should_stop:
            break

        current_iteration += 1
    return response


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

    response = await run(prepared_input)
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
    response = await run(prepared_input, previous_response_id)
    logger.info(f"Second response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
