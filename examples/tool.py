import asyncio
import os
from typing import Callable
import logging

# Configure logging
from openai import AsyncOpenAI
from openai.types import Reasoning
from openai.types.responses import (
    FunctionTool,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseReasoningItem,
    EasyInputMessage,
)

from openai.types.responses.response_input_item_param import FunctionCallOutput
from pydantic import BaseModel

from src.tool import ToolManager

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global tool manager instance
tool_manager = ToolManager()


def function_tool(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> FunctionTool:
    """Wraps a function to be used as a tool."""
    return tool_manager.register_function(func, name, description, strict)


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
def get_weather(city: str) -> Weather:
    """Get the current weather information for a specified city."""
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


async def main():
    """Tool example of agentic system."""

    input_message = "What is weather in Kathmandu?"

    prepared_input = [
        EasyInputMessage(content=input_message, role="user", type="message")
    ]

    # test decorator
    print(get_weather)

    max_iterations = 10
    current_iteration = 0
    agent_should_stop = False
    while True:
        if current_iteration >= max_iterations:
            logger.info("Max iterations reached. Exiting.")
            break

        response = await async_client.responses.create(
            model="gpt-5-nano",
            input=prepared_input,
            tools=tool_manager.tools,
            instructions="You are a helpful assistant.",
            reasoning=Reasoning(
                summary="auto",
            ),
        )

        logger.info(f"Response: {response}")
        breakpoint()

        for output in response.output:
            if isinstance(output, ResponseReasoningItem):
                if output.summary:
                    for summary_item in output.summary:
                        logger.info(f"Reasoning summary: {summary_item.text}")
                else:
                    logger.info("Reasoning summary: None")
            if isinstance(output, ResponseOutputMessage):
                if isinstance(output.content[0], ResponseOutputText):
                    logger.info(f"Final response: {output.content[0].text}")
                    agent_should_stop = True
                else:
                    logger.warning(f"Unsupported output type: {output.content[0]}")
            elif isinstance(output, ResponseFunctionToolCall):
                # tool call should be first in prepared input
                logger.info(f"Tool call: {output}")
                prepared_input.append(
                    ResponseFunctionToolCallParam(
                        call_id=output.call_id,
                        name=output.name,
                        arguments=output.arguments,
                        type="function_call",
                    ),
                )
                # execute the tool call using tool manager
                tool_response = tool_manager.execute_function(
                    output.name, output.arguments
                )
                logger.info(f"Tool response: {tool_response}")
                prepared_input.append(
                    FunctionCallOutput(
                        call_id=output.call_id,
                        output=tool_response,
                        type="function_call_output",
                    ),
                )
                current_iteration += 1

        if agent_should_stop:
            break

    breakpoint()


if __name__ == "__main__":
    asyncio.run(main())
