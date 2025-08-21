import logging
import os
from openai import AsyncOpenAI
from openai.types import Reasoning
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_text_config_param import ResponseTextConfigParam

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def run(
    input: ResponseInputParam,
    previous_response_id: str | None = None,
    max_iterations: int = 10,
    output_type: ResponseTextConfigParam | None = None,
) -> tuple[Response, str | None]:
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
            text=output_type,
        )

        previous_response_id = response.id
        final_output = None

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
                    final_output = output.content[0].text
                    agent_should_stop = True
                else:
                    logger.warning(f"Unsupported output type: {output.content[0]}")
            else:
                logger.warning(f"Unsupported output type: {output}")

        breakpoint()
        if agent_should_stop:
            break

        current_iteration += 1
    return response, final_output
