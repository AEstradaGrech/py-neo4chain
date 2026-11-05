
import json
from typing import Annotated, Any, Dict, List
from fastapi import Request
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from ollama import ResponseError
from typing_extensions import NotRequired
from api.models.kgraph_schemas import KGStepResponseDto
from api.utils.helpers import HTTPLoggedException
from enum import StrEnum
from langchain.agents.middleware import ModelRequest, ModelRetryMiddleware, wrap_model_call

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langfuse.langchain import CallbackHandler
from loguru import logger

from api.infrastructure.llm.statics import ALL_AVAILABLE_FILES, APPROVED_CONSTRUCTION_PLAN, APPROVED_ENTITIES, APPROVED_FACTS, APPROVED_FILES, APPROVED_USER_GOAL, NUDGE_MESSAGE_NAME, PERCEIVED_USER_GOAL, PROPOSED_CONSTRUCTION_PLAN, PROPOSED_ENTITIES, PROPOSED_FACTS, SUGGESTED_FILES, default_llm_model, default_embedding_model, default_embedding_dimensions, default_embedding_numctx, default_agent_numctx

def ollama(model:str = None, temp:float = 0.7, max_tokens:int = 600, ctx_len:int = 4096, verbose:bool = False) -> ChatOllama:
    logger.warning(f"-- OLLAMA INSTANCE FOR: {default_llm_model if model is None else model}")
    return ChatOllama(model=default_llm_model if model is None else model, temperature=temp, num_predict=max_tokens, num_ctx=ctx_len, verbose=verbose)

def agent_chat_model(model: str = None, temperature:float = 0.7, top_p:float = 0.9, top_k:int=40, num_predict:int = -2, ctx_len: int = default_agent_numctx) -> ChatOllama:
    """Chat model for agents. 'ollama:<model>' strings use Ollama's default 4096 context: the prompt overflows it and the answer gets cut"""
    return ChatOllama(
        model=default_llm_model if model is None else model, 
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        num_ctx=ctx_len)

def ollama_embeddings(dimensions:int = None, model:str = None, num_ctx:int = None) -> OllamaEmbeddings:
    return  OllamaEmbeddings(
        model=default_embedding_model if model is None else model,
        dimensions=default_embedding_dimensions if dimensions is None else dimensions,
        num_ctx=default_embedding_numctx if num_ctx is None else num_ctx 
    )
def tracing_config(trace_name: str, tags: List[str] = None) -> Dict[str, Any]:
    """RunnableConfig that sends the run to Langfuse (keys from .env). Empty config (no tracing) if LANGFUSE_PUBLIC_KEY isn't set"""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return {}
    return {
        "callbacks": [CallbackHandler()],
        "metadata": {"langfuse_trace_name": trace_name, "langfuse_tags": tags if tags is not None else []}
    }

def kgraph_tracing(step_tag:str) -> Dict[str, Any]:
    return tracing_config("kgraph", ["kgraph", step_tag])

def merge_dicts(current: Dict[str, Any] | None, update: Dict[str, Any] | None) -> Dict[str, Any]:
    """Reducer: merges the update into the current dict. A None value removes that key"""
    merged = {**(current or {}), **(update or {})}
    return {k: v for k, v in merged.items() if v is not None}

class KGState(AgentState):
    """Agent state shared between tool calls.
    Pass it as create_agent(state_schema=KGState): Command updates on keys not declared here are silently dropped"""
    perceived_user_goal: NotRequired[Dict[str, Any]]
    approved_user_goal: NotRequired[Dict[str, Any]]
    all_available_files: NotRequired[List[str]]
    suggested_files: NotRequired[List[str]]
    approved_files: NotRequired[List[str]]
    # merge_dicts reducer: tools write only the entries they add ({key: rule}) or remove ({key: None})
    proposed_construction_plan: NotRequired[Annotated[Dict[str, Any], merge_dicts]]
    approved_construction_plan: NotRequired[Dict[str, Any]]
    # schema critic output ('valid' or 'retry' + problems), read by the schema proposal agent prompt
    feedback: NotRequired[str]
    inspected_ner_files: NotRequired[List[str]]
    proposed_entity_types: NotRequired[List[str]]
    approved_entity_types: NotRequired[List[str]]
    # merge_dicts reducer: add_proposed_fact writes only the fact it adds (the LLM proposes several facts in parallel)
    proposed_fact_types: NotRequired[Annotated[Dict[str, Any], merge_dicts]]
    approved_fact_types: NotRequired[Dict[str, Any]]

def dict_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Changes from `before` to `after` as a merge_dicts update: new/changed entries, and removed ones as None.
    Writing `after` as is to a merge_dicts key would merge the removed entries back"""
    return {**{k: None for k in before if k not in after}, **{k: v for k, v in after.items() if before.get(k) != v}}

def tool_success(key: str, result: Any) -> Dict[str, Any]:
    return {"status": "success", key: result}

def tool_error(message: str) -> Dict[str, Any]:
    return {"status": "error", "error_message": message}

def state_update(tool_runtime: ToolRuntime, updates: Dict[str, Any], result: Dict[str, Any]) -> Command:
    """Writes `updates` in the agent state and answers the tool call with `result` (what the LLM sees)"""
    return Command(update={
        **updates,
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_runtime.tool_call_id)],
    })

def update_state(tool_runtime: ToolRuntime, key: str, value: Any) -> Command:
    """Writes key=value in the agent state and answers the tool call.
    Assigning tool_runtime.state[key] directly fails: the state only changes through what a tool returns"""
    return Command(update={
        key: value,
        # a tool returning a Command must also return the ToolMessage that closes its tool call
        "messages": [ToolMessage(content=json.dumps(tool_success(key, value)), tool_call_id=tool_runtime.tool_call_id)],
    })

async def get_agent_messages(agent, config):
    return (await agent.aget_state(config)).values["messages"]

async def get_agent_state(agent, thread_id):
    state = await agent.aget_state(thread_config(thread_id))
    return state.values

def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

def agent_config(thread_id: str, tracing: Dict[str, Any]) -> Dict[str, Any]:
    """Tracing config + thread_id for the checkpointer (and as Langfuse session, to group the conversation traces)"""
    config = {**tracing, "configurable": {"thread_id": thread_id}}
    if "metadata" in config:
        config["metadata"] = {**config["metadata"], "langfuse_session_id": thread_id}
    return config

async def get_thread_values(request: Request, thread_id: str) -> dict:
    checkpoint = await request.app.state.neo4j_saver.aget_tuple(thread_config(thread_id))
    if checkpoint is None:
        raise HTTPLoggedException(status_code=404, detail=f"Thread '{thread_id}' not found")
    # skip LangGraph internal channels (__pregel_tasks, branch:to:model...)
    return {k: v for k, v in checkpoint.checkpoint["channel_values"].items() if not k.startswith("__") and ":" not in k}


class StepKey(StrEnum):
    PERCEIVED_USER_GOAL = "perceived_user_goal"
    SUGGESTED_FILES = "suggested_files"
    PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
    SCHEMA_CRITIC = "schema_critic"
    SCHEMA_REFINER = "schema_refiner"
    PROPOSED_ENTITIES = "proposed_entity_types"
    PROPOSED_FACTS = "proposed_fact_types"

def step_response(name:str, response, approved:bool, thread_id:str):
    match name:
        case StepKey.PERCEIVED_USER_GOAL:
            if approved:
                return KGStepResponseDto(
                    thread_id=thread_id,
                    state_key=PERCEIVED_USER_GOAL,
                    status="perceived_goal_approved" if APPROVED_USER_GOAL in response else "needs_clarification",
                    message=response["messages"][-1].content,
                    output_state={
                        PERCEIVED_USER_GOAL: response.get(PERCEIVED_USER_GOAL),
                        APPROVED_USER_GOAL: response.get(APPROVED_USER_GOAL),
                })
            return KGStepResponseDto(
                thread_id=thread_id,
                state_key=PERCEIVED_USER_GOAL,
                status="perceived_goal_approved" if PERCEIVED_USER_GOAL in response else "needs_clarification",
                message=response["messages"][-1].content,
                output_state={
                    PERCEIVED_USER_GOAL: response.get(PERCEIVED_USER_GOAL),
            })
        case StepKey.SUGGESTED_FILES:
            if approved:
                return KGStepResponseDto(
                    thread_id=thread_id,
                    state_key=SUGGESTED_FILES,
                    status="files_approved" if APPROVED_FILES in response else "file_approval_failure",
                    message=response["messages"][-1].content,
                    output_state={
                        APPROVED_FILES: response.get(APPROVED_FILES),
                        SUGGESTED_FILES: response.get(SUGGESTED_FILES),
                        ALL_AVAILABLE_FILES: response.get(ALL_AVAILABLE_FILES),
                })
            return KGStepResponseDto(
                thread_id=thread_id,
                state_key=SUGGESTED_FILES,
                status="files_suggested" if SUGGESTED_FILES in response else "needs_clarification",
                message=response["messages"][-1].content,
                output_state={
                    SUGGESTED_FILES: response.get(SUGGESTED_FILES),
                    ALL_AVAILABLE_FILES: response.get(ALL_AVAILABLE_FILES),
                })
        case StepKey.PROPOSED_CONSTRUCTION_PLAN:
            if approved:
                return KGStepResponseDto(
                    thread_id=thread_id,
                    state_key=PROPOSED_CONSTRUCTION_PLAN,
                    status="approved_graph_schema" if response.get(APPROVED_CONSTRUCTION_PLAN) else "graph_approval_error",
                    message=response["messages"][-1].content,
                    output_state={
                        PROPOSED_CONSTRUCTION_PLAN: response.get(PROPOSED_CONSTRUCTION_PLAN),
                        APPROVED_CONSTRUCTION_PLAN: response.get(APPROVED_CONSTRUCTION_PLAN)
                    })
            return KGStepResponseDto(
                thread_id=thread_id,
                state_key=PROPOSED_CONSTRUCTION_PLAN,
                status="graph_schema_proposal" if response.get(PROPOSED_CONSTRUCTION_PLAN) else "graph_construction_error",
                message=response["messages"][-1].content,
                output_state={
                    PROPOSED_CONSTRUCTION_PLAN: response.get(PROPOSED_CONSTRUCTION_PLAN)
                })
        case StepKey.PROPOSED_ENTITIES:
            if approved:
                return KGStepResponseDto(
                    thread_id=thread_id,
                    state_key=PROPOSED_ENTITIES,
                    status="approved_ner_schema" if response.get(APPROVED_ENTITIES) else "ner_approval_error",
                    message=response["messages"][-1].content,
                    output_state={
                        PROPOSED_ENTITIES: response.get(PROPOSED_ENTITIES),
                        APPROVED_ENTITIES: response.get(APPROVED_ENTITIES)
                    })
            return KGStepResponseDto(
                thread_id=thread_id,
                state_key=PROPOSED_ENTITIES,
                status="generated_ner_schema" if response.get(PROPOSED_ENTITIES) else "ner_schema_error",
                message=response["messages"][-1].content,
                output_state={
                    PROPOSED_ENTITIES: response.get(PROPOSED_ENTITIES)
                })
        case StepKey.PROPOSED_FACTS:
            if approved:
                return KGStepResponseDto(
                    thread_id=thread_id,
                    state_key=PROPOSED_FACTS,
                    status="approved_facts_schema" if response.get(APPROVED_FACTS) else "facts_approval_error",
                    message=response["messages"][-1].content,
                    output_state={
                        PROPOSED_FACTS: response.get(PROPOSED_FACTS),
                        APPROVED_FACTS: response.get(APPROVED_FACTS)
                    })
            return KGStepResponseDto(
                thread_id=thread_id,
                state_key=PROPOSED_FACTS,
                status="generated_facts_schema" if response.get(PROPOSED_FACTS) else "facts_schema_error",
                message=response["messages"][-1].content,
                output_state={
                    PROPOSED_FACTS: response.get(PROPOSED_FACTS)
                })

def ollama_retry_middleware(retries:int = 1, initial_delay:float=0.5):
    if retries <= 0:
        retries = 1
    if initial_delay < 0:
        initial_delay = 0
    return ModelRetryMiddleware(max_retries=retries, retry_on=(ResponseError,), initial_delay=initial_delay)


def nudge_message(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content, "name": NUDGE_MESSAGE_NAME}

@wrap_model_call
async def current_request_only(request: ModelRequest, handler):
    """The model only gets the messages of the current request: the thread keeps the full history,
    and the agent reads what it needs from previous steps through its tools (state keys).
    Nudge messages don't start a new request, so the model keeps seeing the work it did before the nudge"""
    last_human = max((i for i, m in enumerate(request.messages)
                      if isinstance(m, HumanMessage) and m.name != NUDGE_MESSAGE_NAME), default=0)
    return await handler(request.override(messages=request.messages[last_human:]))