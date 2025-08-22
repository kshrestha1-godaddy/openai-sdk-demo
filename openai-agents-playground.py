import os
from dotenv import load_dotenv
import asyncio
from typing import Any
load_dotenv()
from agents import set_default_openai_key
set_default_openai_key(os.environ["OPENAI_API_KEY"])

from agents import Agent, ModelSettings, function_tool


from agents import Agent, RunContextWrapper, RunHooks, Runner, Tool, Usage, function_tool

class MyHooks(RunHooks):
    
    def __init__(self):
        self.event_counter = 0
        # added - keep track of total input and output tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0 

    def _usage_to_str(self, usage: Usage) -> str:
        return f"{usage.requests} requests, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, {usage.total_tokens} total tokens"

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1                         
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        print(
        f"### {self.event_counter}: Agent {agent.name} started. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        
        print(
        f"### {self.event_counter}: Agent {agent.name} ended with output {output}. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        print(
        f"### {self.event_counter}: Tool {tool.name} started. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
       
        print(
        f"### {self.event_counter}: Tool {tool.name} ended with result {result}. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        print(
        f"### {self.event_counter}: Handoff from {from_agent.name} to {to_agent.name}. Usage: {self._usage_to_str(context.usage)}"
        ) 

import random
from pydantic import BaseModel

myhooks = MyHooks()

@function_tool
def random_number(max: int) -> int:
    """Generate a random number up to the provided max."""
    return random.randint(0, max)


@function_tool
def multiply_by_two(x: int) -> int:
    """Return x times two."""
    return x * 2


class FinalResult(BaseModel):
    number: int


multiply_agent = Agent(
    name="Multiply Agent",
    instructions="Multiply the number by 2 and then return the final result.",
    tools=[multiply_by_two],
    output_type=FinalResult,
)

start_agent = Agent(
    name="Start Agent",
    instructions="Generate a random number. If it's even, stop. If it's odd, hand off to the multipler agent.",
    tools=[random_number],
    output_type=FinalResult,
    handoffs=[multiply_agent],
)

async def main() -> None:
    
    max_number = 99
    result = Runner.run_streamed(
        start_agent,
        hooks=myhooks,
        input=f"Generate a random number between 0 and {max_number}.",
    )


    async for event in result.stream_events():
        print(event)


    print(f"\n Final Output: {result.final_output}")
    print(f"\n Total Input Tokens: {myhooks.total_input_tokens} Total Output Tokens: {myhooks.total_output_tokens}")

if __name__ == "__main__":
    asyncio.run(main())