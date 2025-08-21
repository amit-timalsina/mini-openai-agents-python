"""
NOTE: This example will not work out of the box, because the default prompt ID will not be available
in your project.

To use it, please:
1. Go to https://platform.openai.com/playground/prompts
2. Create a new prompt variable, `poem_style`.
3. Create a system prompt with the content:
```
Write a poem in {{poem_style}}
```
"""

import os
import logging
import asyncio
from openai import AsyncOpenAI
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_param import Message
from openai.types.responses import (
    ResponsePromptParam,
    ResponseReasoningItem,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types import Reasoning

PROMPT_ID = "pmpt_68a698cbc4e08196860107b4f2d318a10ed4ffeb0cc3bad6"


async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Example of using a prompt template support in OpenAI playground."""

    prepared_input: ResponseInputParam = [
        Message(
            role="user",
            content=[
                ResponseInputTextParam(
                    type="input_text", text="Tell me about recursion in programming."
                )
            ],
        )
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
            prompt=ResponsePromptParam(
                id=PROMPT_ID,
                version="1",
                variables={
                    "poem_style": "Haiku",
                },
            ),
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
