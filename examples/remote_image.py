"""
This example shows how to use the OpenAI SDK to provide remote image and use it as a context for a chat.
"""

import asyncio
import os
import logging
from openai import AsyncOpenAI
from openai.types import Reasoning
from openai.types.responses.response_input_param import ResponseInputParam, Message
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def main() -> None:
    """
    Main function to provide remote image and use it as a context for a chat.
    """

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
    )

    prepared_input: ResponseInputParam = [
        Message(
            role="user",
            content=[
                ResponseInputImageParam(
                    type="input_image",
                    detail="auto",
                    image_url=image_url,
                )
            ],
        ),
        Message(
            role="user",
            content=[
                ResponseInputTextParam(
                    type="input_text", text="What do you see in the image?"
                )
            ],
        ),
    ]

    current_iteration = 0
    max_iterations = 10
    agent_should_stop = False

    while True:
        if current_iteration >= max_iterations:
            logger.info("Max iterations reached. Exiting.")
            break

        response = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
            model="gpt-5-nano",
            input=prepared_input,
            reasoning=Reasoning(summary="auto"),
            instructions="You are a helpful assistant.",
        )

        logger.info(f"Response: {response}")

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

        prepared_input.extend(response.output)


if __name__ == "__main__":
    asyncio.run(main())
