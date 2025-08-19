import asyncio
import os
import json

from typing import Any, Callable
from openai import AsyncOpenAI
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseTextDeltaEvent,
    ResponseContentPartDoneEvent,
)
from openai.types import Reasoning

from src.tool import ToolManager
from openai.types.responses import (
    FunctionTool,
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
def write_to_file(file_path: str, content: str) -> str:
    """Writes to a file."""
    with open(file_path, "w") as f:
        f.write(content)

    return f"File {file_path} written successfully."


@function_tool
def create_config(project_name: str, version: str, dependencies: list[str]) -> str:
    """Creates a config file."""
    with open(f"{project_name}.json", "w") as f:
        json.dump(
            {
                "project_name": project_name,
                "version": version,
                "dependencies": dependencies,
            },
            f,
        )

    return f"Config file {project_name}.json created successfully."


async def main() -> None:
    """Stream function args example with continuous user interaction.

    This interactive agent allows users to:
    - Start a conversation with an initial message
    - Continue the conversation after each assistant response
    - Use tools (write_to_file, create_config) through the assistant
    - Exit the conversation with commands like 'exit', 'quit', 'bye', 'stop'
    """

    print("🤖 AI Assistant with Tool Support")
    print("═" * 40)
    print(
        "You can ask me to create files, configurations, or help with development tasks."
    )
    print("Type 'exit', 'quit', 'bye', or 'stop' to end the conversation.")
    print("═" * 40)

    # Get initial input from user
    input_text = "Create a Python web project called 'my-app' with FastAPI. Version 1.0.0, dependencies: fastapi, uvicorn"

    prepared_input: ResponseInputParam = [
        EasyInputMessageParam(content=input_text, role="user", type="message")
    ]
    # creating a directory to write the file to
    os.makedirs("my-app", exist_ok=True)

    current_iteration = 0
    max_iterations = 10  # Increased to allow more back-and-forth
    agent_should_stop = False

    function_calls: dict[str, dict[str, Any]] = {}  # call_id -> {name, arguments}
    current_active_call_id = None

    while True:
        if current_iteration >= max_iterations:
            print(
                f"\n⚠️ Maximum iterations ({max_iterations}) reached. Conversation ended."
            )
            break

        response = await async_client.responses.create(  # type: ignore[call-overload] # Not sure why mypy is complaining
            model="gpt-5-nano",
            input=prepared_input,
            instructions="You are a helpful coding assistant. Use the provided tools to create files and configurations",
            reasoning=Reasoning(
                summary="auto",
            ),
            stream=True,
            tools=tool_manager.tools,
        )

        async for chunk in response:
            if isinstance(chunk, ResponseReasoningSummaryPartAddedEvent):
                logger.info("Reasoning Summary Part Added")
            elif isinstance(chunk, ResponseReasoningSummaryTextDeltaEvent):
                print(chunk.delta, end="", flush=True)
            elif isinstance(chunk, ResponseReasoningSummaryTextDoneEvent):
                logger.info("Reasoning Summary Text Done")
            elif isinstance(chunk, ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, ResponseFunctionToolCall):
                    function_name = chunk.item.name
                    call_id = chunk.item.call_id

                    function_calls[call_id] = {"name": function_name, "arguments": ""}
                    current_active_call_id = call_id
                    print(f"\n📞 Function call streaming started: {function_name}()")
                    print("📝 Arguments building...")
            elif isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                if current_active_call_id and current_active_call_id in function_calls:
                    function_calls[current_active_call_id]["arguments"] += chunk.delta
                    print(chunk.delta, end="", flush=True)
            elif isinstance(chunk, ResponseOutputItemDoneEvent):
                if isinstance(chunk.item, ResponseFunctionToolCall):
                    call_id = chunk.item.call_id
                    if call_id in function_calls:
                        function_info = function_calls[call_id]
                        print(
                            f"\n✅ Function call streaming completed: {function_info['name']}\n\n"
                        )
                        # Execute the function
                        tool_response = tool_manager.execute_function(
                            function_info["name"], function_info["arguments"]
                        )
                        print(f"🔧 Tool response: {tool_response}\n")

                        # Add tool call and response to prepared input for conversation continuity
                        prepared_input.append(
                            ResponseFunctionToolCallParam(
                                call_id=call_id,
                                name=function_info["name"],
                                arguments=function_info["arguments"],
                                type="function_call",
                            )
                        )
                        prepared_input.append(
                            FunctionCallOutput(
                                call_id=call_id,
                                output=tool_response,
                                type="function_call_output",
                            )
                        )

                        if current_active_call_id == call_id:
                            current_active_call_id = None
            elif isinstance(chunk, ResponseContentPartAddedEvent):
                print("Assistant: ", end="", flush=True)
            elif isinstance(chunk, ResponseTextDeltaEvent):
                print(chunk.delta, end="", flush=True)
            elif isinstance(chunk, ResponseContentPartDoneEvent):
                print("\n")
                agent_should_stop = True

        current_iteration += 1

        if agent_should_stop:
            # Ask user for follow-up input
            user_input = input("\nYou: ").strip()

            # If user wants to end conversation
            if user_input.lower() in ["exit", "quit", "bye", "stop"]:
                print("Assistant: Goodbye! 👋")
                break

            # If user provides new input, continue conversation
            if user_input:
                prepared_input.append(
                    EasyInputMessageParam(
                        content=user_input, role="user", type="message"
                    )
                )
                agent_should_stop = False
            else:
                print("Assistant: Goodbye! 👋")
                break


if __name__ == "__main__":
    asyncio.run(main())
