import asyncio
import logging as logger
import os
from openai import AsyncOpenAI
from openai.types import Reasoning
from openai.types.responses import ResponseOutputMessage

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def main():
    """Hello world example of agentic system."""
    input_text = "What is the capital of France?"

    max_iterations = 10
    current_iteration = 0
    agent_should_stop = False
    while True:
        if current_iteration >= max_iterations:
            logger.info("Max iterations reached. Exiting.")
            break

        response = await async_client.responses.create(
            model="gpt-5-nano",
            input=input_text,
            temperature=1.0,
            reasoning=Reasoning(
                summary="auto",
            ),
            instructions="You are a helpful assistant.",
        )
        
        logger.info(f"Response: {response}")
        breakpoint()

        # if output is of type string. then we have reached final response.
        for output in response.output:
            logger.info(f"Output: {output}")
            if isinstance(output, ResponseOutputMessage):
                if output.content[0].type == "output_text":
                    logger.info(f"Final response: {output.content[0].text}")
                    agent_should_stop = True
                else:
                    logger.warning(f"Unsupported output type: {output.content[0]}")
                
            else:
                current_iteration += 1
        
        if agent_should_stop:
            break


if __name__ == "__main__":
    asyncio.run(main())