import json
import os
from typing import Callable
import logging

# Configure logging
from openai import AsyncOpenAI
from openai.types.responses import (
    FunctionTool,
)


async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ToolManager:
    """Manages function tools and their execution."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}
        self._tools: list[FunctionTool] = []

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
    ) -> FunctionTool:
        """Register a function as a tool."""
        # Convert Python type annotations to JSON schema format
        annotations = func.__annotations__.copy()
        annotations.pop("return", None)

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
        return json.dumps(func(**args))

    @property
    def tools(self) -> list[FunctionTool]:
        """Get all registered tools."""
        return self._tools.copy()
