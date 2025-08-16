import asyncio
import json
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
)

from openai.types.responses.response_input_item_param import Message
from openai.types.responses.response_input_item_param import FunctionCallOutput

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ToolManager:
    """Manages function tools and their execution."""
    
    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}
        self._tools: list[FunctionTool] = []
    
    def register_function(self, func: Callable, name: str | None = None, description: str | None = None, strict: bool = False) -> FunctionTool:
        """Register a function as a tool."""
        # Convert Python type annotations to JSON schema format
        annotations = func.__annotations__.copy()
        annotations.pop('return', None)
        
        properties = {}
        required = []
        
        for param_name, param_type in annotations.items():
            if param_type is str:
                properties[param_name] = {"type": "string"}
            elif param_type is int:
                properties[param_name] = {"type": "integer"}
            elif param_type is float:
                properties[param_name] = {"type": "number"}
            elif param_type is bool:
                properties[param_name] = {"type": "boolean"}
            elif param_type is list:
                properties[param_name] = {"type": "array"}
            elif param_type is dict:
                properties[param_name] = {"type": "object"}
            else:
                properties[param_name] = {"type": "string"}
            
            required.append(param_name)
        
        parameters: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        
        function_name = name or func.__name__
        self._functions[function_name] = func
        
        tool = FunctionTool(
            name=function_name,
            description=description or func.__doc__,
            parameters=parameters,
            type="function",
            strict=strict,
        )
        
        self._tools.append(tool)
        return tool
    
    def execute_function(self, name: str, arguments: str) -> str:
        """Execute a registered function by name with JSON arguments."""
        if name not in self._functions:
            raise ValueError(f"Function '{name}' not found in registry")
        
        args = json.loads(arguments)
        func = self._functions[name]
        return func(**args)
    
    @property
    def tools(self) -> list[FunctionTool]:
        """Get all registered tools."""
        return self._tools.copy()


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


@function_tool
def get_weather(city: str) -> str:
    """Get the weather of a city."""
    return f"The weather of {city} is sunny."


async def main():
    """Tool example of agentic system."""

    input_message = "What is weather in Kathmandu?"

    prepared_input = [Message(content=input_message, role="user", type="message")]

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
                tool_response = tool_manager.execute_function(output.name, output.arguments)
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
