import asyncio
import os

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from dotenv import load_dotenv

load_dotenv()

from agents import set_default_openai_key
set_default_openai_key(os.environ["OPENAI_API_KEY"])


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to answer the questions.",
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="required"),
    )

    message = "Get the values of a and b and find their sum"

    print(f"\n\nRunning: {message}\n\n")

    result = Runner.run_streamed(
        agent,
        input=message,
    )

    async for event in result.stream_events():
        print(event)


    print(result.final_output)


async def main():
    async with MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": "http://127.0.0.1:8000/mcp/"
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="Streamable HTTP Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)


if __name__ == "__main__":

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        exit(1)