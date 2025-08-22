import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env")


from typing import Any

from agents import set_default_openai_key

set_default_openai_key(os.environ["OPENAI_API_KEY"])


import mlflow
# mlflow.openai.autolog()
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("OpenAI Agent 2")


###################################
###################################

import asyncio
from agents import Agent, Runner, function_tool, RunContextWrapper, Session, RunConfig

from typing import List


class MyCustomSession(Session):
    """Custom session implementation following the Session protocol."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.items = []
        # Your initialization here

    async def get_items(self, limit: int | None = None) -> List[dict]:
        """Retrieve conversation history for this session."""
        # Your implementation here
        return self.items

    async def add_items(self, items: List[dict]) -> None:
        """Store new items for this session."""
        # Your implementation here
        self.items.extend(items)

    async def pop_item(self) -> dict | None:
        """Remove and return the most recent item from this session."""
        # Your implementation here
        return self.items.pop()

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        # Your implementation here
        self.items = []


from agents import Agent, RunContextWrapper, Runner, Usage, RunConfig
from tracing.mlflow_tracer import MLFlowTracerHooks



from agents.run import (
    ModelSettings,
    CallModelData,
    ModelInputData,
)


class InputFilter:
    def __init__(self, **args):
        self.args = args

    def __call__(self, call_model_data: CallModelData):
        print(call_model_data)

        return ModelInputData(
            input=call_model_data.model_data.input,
            instructions=call_model_data.model_data.instructions,
        )


@function_tool
def add_two_numbers(a: int, b: int) -> int:
    return a + b


from pydantic import BaseModel
from datetime import datetime


class CalenderEvent(BaseModel):
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    participants: list[str]


@dataclass
class UserContext:
    name: str


def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."


run_config = RunConfig(
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ),
    tracing_disabled=False,
    workflow_name="my_workflow",
    trace_id="trace_my_trace_id",
    call_model_input_filter=InputFilter(),
)


async def main():
    user_query = "I have a meeting on Himalayn Java with the CEO of Google, on Friday 3pm. Give me the sum of 10 and 9 as well using the tool add_two_numbers"



    hooks = MLFlowTracerHooks(
        run_name="agent_run",
        description="This is a test run",
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="OpenAI Agent 2",
        tags={"workflow": "my_workflow"},
        run_config=run_config,
        request_preview=user_query,
    )

    agent = Agent[UserContext](
        name="Calendar extractor",
        instructions=dynamic_instructions,
        tools=[add_two_numbers],
        output_type=CalenderEvent,
    )

    result = Runner.run_streamed(
        agent,
        user_query,
        session=MyCustomSession("my_session"),
        context=UserContext(name="Kamal"),
        run_config=run_config,
        hooks=hooks,
    )


    async for event in result.stream_events():

        print(event)

        hooks.handle_stream_event(event)

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
