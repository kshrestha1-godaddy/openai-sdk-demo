import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, Optional
from agents import Agent, RunContextWrapper, Usage, RunConfig, RunHooks, Tool

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TokenUsageKey,
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
)
from mlflow.tracing.fluent import start_span_no_context



class MLFlowTracerHooks(RunHooks):
    """Comprehensive MLflow tracer and hooks for agent workflows.
    
    This class combines MLflow run management with agent lifecycle hooks, providing:
    - Complete MLflow run lifecycle management
    - Parameters and system metrics from run configuration
    - Comprehensive span tracking with timeline details
    - Token usage metrics and execution timing
    - Detailed stream event processing
    - Artifact and event logging
    """
    
    def __init__(
        self,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        run_config: RunConfig | None = None,
        request_preview: str | None = None,
    ):
        # MLflow setup
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        # Start the MLflow run
        self._run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self._run.info.run_id
        
        # Set description if provided
        if description:
            self.set_description(description)

        # Temp dir to buffer artifacts before uploading
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mlflow-tracer-")
        self._events_path = os.path.join(self._tmpdir.name, "events.jsonl")
        # Ensure file exists
        with open(self._events_path, "a", encoding="utf-8"):
            pass

        # Agent hooks configuration
        self.run_config = run_config
        self.event_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.root_span = None
        self.current_tool_span = None
        self._current_request = request_preview
        self._config_logged = False
        self._stream_index = 0
        self._params_logged = False
        self._start_time = None
        self._end_time = None

    # ---------- MLflow API Methods ----------
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        safe_params = {k: self._stringify(v) for k, v in params.items()}
        mlflow.log_params(safe_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_event(self, event: Dict[str, Any]) -> None:
        """Log an event to the events.jsonl file."""
        enriched = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **event,
        }
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, default=self._json_default))
            f.write("\n")

    def log_text(self, artifact_path: str, content: str) -> None:
        """Log text content as an artifact."""
        full_path = os.path.join(self._tmpdir.name, artifact_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    def set_description(self, description: str) -> None:
        """Set the description for the current MLflow run."""
        try:
            mlflow.set_tag("mlflow.note.content", description)
        except Exception:
            pass

    def end_run(self, status: str = "FINISHED") -> None:
        """End the MLflow run and upload artifacts."""
        # Flush events.jsonl
        mlflow.log_artifact(self._events_path, artifact_path="trace")
        # Upload any other files created in tmpdir
        for root, _dirs, files in os.walk(self._tmpdir.name):
            for name in files:
                full = os.path.join(root, name)
                if full == self._events_path:
                    continue  # already uploaded under trace/
                rel = os.path.relpath(full, self._tmpdir.name)
                mlflow.log_artifact(full, artifact_path=os.path.dirname(rel) or None)

        mlflow.end_run(status=status)
        self._tmpdir.cleanup()

    @staticmethod
    def _stringify(value: Any) -> str:
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return str(value)
            return json.dumps(value, default=MLFlowTracerHooks._json_default)
        except Exception:
            return str(value)

    @staticmethod
    def _json_default(obj: Any) -> str:
        # Fallback serializer for non-JSON-serializable objects
        try:
            return getattr(obj, "model_dump_json", None) and obj.model_dump_json()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return getattr(obj, "json", None) and obj.json()  # type: ignore[call-arg]
        except Exception:
            pass
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass
        return str(obj)

    def _usage_to_dict(self, usage: Usage) -> dict:
        return {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }


    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self._start_time = time.time()
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens

        # Log run params derived from run_config once
        if not self._params_logged:
            try:
                params = {}
                if self.run_config:
                    # Log all model-related settings as parameters
                    if getattr(self.run_config, "model", None):
                        params["model"] = str(self.run_config.model)
                    if getattr(self.run_config, "workflow_name", None):
                        params["workflow_name"] = str(self.run_config.workflow_name)
                    if getattr(self.run_config, "group_id", None):
                        params["group_id"] = str(self.run_config.group_id)
                    if getattr(self.run_config, "trace_id", None):
                        params["trace_id"] = str(self.run_config.trace_id)
                    if getattr(self.run_config, "tracing_disabled", None) is not None:
                        params["tracing_disabled"] = str(
                            self.run_config.tracing_disabled
                        )

                    # Log all model settings as parameters
                    ms = getattr(self.run_config, "model_settings", None)
                    if ms:
                        # Log all available model settings
                        for attr in [
                            "temperature",
                            "max_tokens",
                            "top_p",
                            "frequency_penalty",
                            "presence_penalty",
                            "stop",
                            "seed",
                            "logit_bias",
                            "user",
                        ]:
                            value = getattr(ms, attr, None)
                            if value is not None:
                                params[f"{attr}"] = value

                    # Log model provider info if available
                    mp = getattr(self.run_config, "model_provider", None)
                    if mp:
                        params["model_provider"] = str(type(mp).__name__)

                if params:
                    self.log_params(params)

                # Add description to the run
                description = f"Agent workflow: {agent.name}"
                if self.run_config and getattr(self.run_config, "workflow_name", None):
                    description += f" - {self.run_config.workflow_name}"
                description += (
                    f" | Request: {self._current_request[:100]}..."
                    if self._current_request and len(self._current_request) > 100
                    else f" | Request: {self._current_request}"
                )

                self.set_description(description)
                self._params_logged = True
            except Exception as e:
                print(f"Error logging params: {e}")
                pass
                
        # start a root span so this appears in Traces UI
        self.root_span = start_span_no_context(
            name=f"agent:{agent.name}",
            span_type=SpanType.AGENT,
            attributes={
                "event": "agent_start",
                SpanAttributeKey.MESSAGE_FORMAT: "openai-agents",
                SpanAttributeKey.CHAT_USAGE: {
                    TokenUsageKey.INPUT_TOKENS: context.usage.input_tokens,
                    TokenUsageKey.OUTPUT_TOKENS: context.usage.output_tokens,
                    TokenUsageKey.TOTAL_TOKENS: context.usage.total_tokens,
                },
            },
            inputs=self._current_request,
        )            
        self.log_event(
            {
                "type": "agent_start",
                "agent": agent.name,
                "usage": self._usage_to_dict(context.usage),
                "event_idx": self.event_counter,
            }
        )

    async def on_agent_end(
        self, context: RunContextWrapper, agent: Agent, output: Any
    ) -> None:
        self._end_time = time.time()
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens

        self.log_event(
            {
                "type": "agent_end",
                "agent": agent.name,
                "output": output,
                "usage": self._usage_to_dict(context.usage),
                "event_idx": self.event_counter,
                "execution_time_seconds": (
                    self._end_time - self._start_time if self._start_time else None
                ),
            }
        )

        # close root span with detailed attributes
        if self.root_span:
            # Add comprehensive span attributes for timeline details
            self.root_span.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        TokenUsageKey.INPUT_TOKENS: self.total_input_tokens,
                        TokenUsageKey.OUTPUT_TOKENS: self.total_output_tokens,
                        TokenUsageKey.TOTAL_TOKENS: self.total_input_tokens
                        + self.total_output_tokens,
                    },
                    "agent.name": agent.name,
                    "agent.output_type": str(getattr(agent, "output_type", None)),
                    "total_events": self.event_counter,
                    "execution_time_seconds": (
                        self._end_time - self._start_time if self._start_time else None
                    ),
                    "workflow_name": (
                        getattr(self.run_config, "workflow_name", None)
                        if self.run_config
                        else None
                    ),
                }
            )
            self.root_span.end(outputs=output)

        # Log final aggregated metrics including system timing
        try:
            final_metrics = {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
            }

            # Log total execution time as system metric
            if self._start_time and self._end_time:
                execution_time = self._end_time - self._start_time
                final_metrics["execution_time_seconds"] = execution_time
                final_metrics["total_events"] = self.event_counter

            self.log_metrics(final_metrics)
            self.log_event({"type": "final_output", "payload": output})
        finally:
            self.end_run()

    async def on_tool_start(
        self, context: RunContextWrapper, agent: Agent, tool: Tool
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens

        # child span under root with detailed attributes
        if self.root_span:
            tool_inputs = {"tool_name": tool.name}
            # Try to get tool function details
            if hasattr(tool, "function"):
                tool_inputs["function_name"] = getattr(
                    tool.function, "__name__", str(tool.function)
                )
                if hasattr(tool.function, "__doc__") and tool.function.__doc__:
                    tool_inputs["function_description"] = tool.function.__doc__.strip()

            self.current_tool_span = start_span_no_context(
                name=f"tool:{tool.name}",
                span_type=SpanType.TOOL,
                parent_span=self.root_span,
                attributes={
                    "event": "tool_start",
                    "agent.name": agent.name,
                    "tool.name": tool.name,
                    "tool.type": str(type(tool).__name__),
                    "event_index": self.event_counter,
                    "timestamp": time.time(),
                    SpanAttributeKey.CHAT_USAGE: {
                        TokenUsageKey.INPUT_TOKENS: context.usage.input_tokens,
                        TokenUsageKey.OUTPUT_TOKENS: context.usage.output_tokens,
                        TokenUsageKey.TOTAL_TOKENS: context.usage.total_tokens,
                    },
                },
                inputs=tool_inputs,
            )
        self.log_event(
            {
                "type": "tool_start",
                "agent": agent.name,
                "tool": tool.name,
                "usage": self._usage_to_dict(context.usage),
                "event_idx": self.event_counter,
            }
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens

        self.log_event(
            {
                "type": "tool_end",
                "agent": agent.name,
                "tool": tool.name,
                "result": (
                    result[:200] + "..." if len(str(result)) > 200 else result
                ),  # Truncate long results
                "result_length": len(str(result)),
                "usage": self._usage_to_dict(context.usage),
                "event_idx": self.event_counter,
                "timestamp": time.time(),
            }
        )

        if getattr(self, "current_tool_span", None):
            # Add detailed attributes before ending
            self.current_tool_span.set_attributes(
                {
                    "tool.result_length": len(str(result)),
                    "tool.execution_completed": True,
                    "event_index": self.event_counter,
                    "timestamp_end": time.time(),
                    SpanAttributeKey.CHAT_USAGE: {
                        TokenUsageKey.INPUT_TOKENS: context.usage.input_tokens,
                        TokenUsageKey.OUTPUT_TOKENS: context.usage.output_tokens,
                        TokenUsageKey.TOTAL_TOKENS: context.usage.total_tokens,
                    },
                }
            )
            self.current_tool_span.end(outputs={"result": result})

    def handle_stream_event(self, event: Any) -> None:
        """Record raw response chunks and semantic run_item events as SpanEvents on root span"""
        if not self.root_span:
            return
        try:
            event_type = getattr(event, "type", None)

            if event_type == "raw_response_event":
                # Add detailed chunk events for timeline
                event_data = getattr(event, "data", None)
                chunk_attributes = {
                    STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(
                        event_data,
                        default=lambda o: getattr(o, "model_dump", lambda: str(o))(),
                    ),
                    "chunk_index": self._stream_index,
                    "timestamp": time.time(),
                    "event_type": "llm_chunk",
                }

                # Add more details if available
                if event_data and hasattr(event_data, "choices") and event_data.choices:
                    choice = event_data.choices[0]
                    if hasattr(choice, "delta") and choice.delta:
                        if hasattr(choice.delta, "content") and choice.delta.content:
                            chunk_attributes["content_length"] = len(
                                choice.delta.content
                            )
                        if hasattr(choice.delta, "role") and choice.delta.role:
                            chunk_attributes["role"] = choice.delta.role

                self.root_span.add_event(
                    SpanEvent(
                        name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(
                            index=self._stream_index
                        ),
                        attributes=chunk_attributes,
                    )
                )
                self._stream_index += 1

            elif event_type == "run_item_stream_event":
                # Add detailed run item events
                item_name = getattr(event, "name", "unknown")
                item = getattr(event, "item", None)

                event_attributes = {
                    "event_type": "run_item",
                    "item_name": item_name,
                    "timestamp": time.time(),
                    "event_index": self.event_counter,
                }

                # Add more context based on item type
                if item:
                    event_attributes["item_type"] = str(type(item).__name__)
                    if hasattr(item, "role"):
                        event_attributes["item_role"] = getattr(item, "role", None)
                    if hasattr(item, "content"):
                        content = getattr(item, "content", None)
                        if content:
                            event_attributes["content_length"] = len(str(content))

                self.root_span.add_event(
                    SpanEvent(
                        name=f"run_item:{item_name}",
                        attributes=event_attributes,
                    )
                )

            elif event_type == "agent_updated_stream_event":
                # Agent handoff events
                new_agent = getattr(event, "new_agent", None)
                agent_name = (
                    getattr(new_agent, "name", "unknown") if new_agent else "unknown"
                )

                self.root_span.add_event(
                    SpanEvent(
                        name=f"agent_handoff:{agent_name}",
                        attributes={
                            "event_type": "agent_handoff",
                            "new_agent_name": agent_name,
                            "timestamp": time.time(),
                            "event_index": self.event_counter,
                        },
                    )
                )
            else:
                # Generic events
                name = event_type or getattr(event, "name", "event")
                self.root_span.add_event(
                    SpanEvent(
                        name=name,
                        attributes={
                            "event_type": "generic",
                            "payload": str(event)[:500],  # Truncate long payloads
                            "timestamp": time.time(),
                        },
                    )
                )
        except Exception as e:
            # Log error but don't fail the execution
            try:
                self.root_span.add_event(
                    SpanEvent(
                        name="stream_event_error",
                        attributes={
                            "error": str(e),
                            "event_type": str(type(event).__name__),
                            "timestamp": time.time(),
                        },
                    )
                )
            except:
                pass

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        self.log_event(
            {
                "type": "handoff",
                "from_agent": from_agent.name,
                "to_agent": to_agent.name,
                "usage": self._usage_to_dict(context.usage),
                "event_idx": self.event_counter,
            }
        )


