import asyncio
import os
import logging
from openai import AsyncOpenAI, AsyncStream
from openai.types import Reasoning
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseStreamEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
)
from openai.types.responses.response_input_param import ResponseInputParam

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Stream text example."""

    input_text = "Please tell me 5 joke relevant for Nepal genz."

    prepared_input: str | ResponseInputParam = [
        EasyInputMessageParam(content=input_text, role="user", type="message")
    ]

    max_iterations = 10
    current_iteration = 0
    agent_should_stop = False
    while True:
        if current_iteration >= max_iterations:
            logger.info("Max iterations reached. Exiting.")
            break

        response: AsyncStream[
            ResponseStreamEvent
        ] = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
            model="gpt-5-nano",
            input=prepared_input,
            instructions="You are a helpful assistant which just tells jokes. No questions asked.",
            reasoning=Reasoning(
                summary="auto",
            ),
            stream=True,
        )

        async for chunk in response:
            if isinstance(chunk, ResponseReasoningSummaryPartAddedEvent):
                print("Starting Reasoning Summary")
            if isinstance(chunk, ResponseReasoningSummaryTextDeltaEvent):
                print(chunk.delta, end="", flush=True)
            elif isinstance(chunk, ResponseReasoningSummaryTextDoneEvent):
                print("\nReasoning Summary Done\n\n")
            elif isinstance(chunk, ResponseContentPartAddedEvent):
                print("Starting Content")
            elif isinstance(chunk, ResponseTextDeltaEvent):
                print(chunk.delta, end="", flush=True)
            elif isinstance(chunk, ResponseContentPartDoneEvent):
                print("\nContent Done\n\n")
                agent_should_stop = True

        if agent_should_stop:
            break


if __name__ == "__main__":
    asyncio.run(main())
