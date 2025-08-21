"""
This example demonstrates how to use an output type that is not in strict mode.

Strict mode allows us to generate valid JSON output, but some schemas are not strict-compatible.

In this example, we define an example that is not strict-compatible, and then we run the agent with
strict_json_schema=False. We also demonstrate a custom output type.

To understand which schemas are strict-compatible, see:
https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#supported-schemas
"""

import asyncio
import json
import logging
from openai.types.responses.response_input_param import Message, ResponseInputParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses import (
    ResponseTextConfigParam,
    ResponseFormatTextJSONSchemaConfigParam,
)
from pydantic import BaseModel
from src.orchestrator import run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Joke(BaseModel):
    jokes: list[str]


async def main() -> None:
    """Example of using a non-strict output type for a structured output."""

    prepared_input: ResponseInputParam = [
        Message(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text="Tell me 3 short jokes")
            ],
        )
    ]

    # let's convert the pydantic model to a json schema
    json_schema = Joke.model_json_schema()
    logger.info(f"JSON schema: {json_schema}")

    output_type = ResponseTextConfigParam(
        format=ResponseFormatTextJSONSchemaConfigParam(
            type="json_schema",
            name="jokes",
            schema=json_schema,
            strict=False,
        ),
    )

    response, final_output = await run(prepared_input, output_type=output_type)
    logger.info(f"Response: {response}")
    logger.info(f"Final output: {final_output}")

    # convert the output to a Joke class using pydantic
    # first conver str to dictionary
    if final_output is None:
        raise ValueError("No output")

    dict_output = json.loads(final_output)

    joke = Joke.model_validate(dict_output)
    logger.info(f"Joke: {joke}")

    breakpoint()


if __name__ == "__main__":
    asyncio.run(main())
