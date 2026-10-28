
import json
from typing import Annotated, Any, Dict, List
from fastapi import Request
from langchain.agents import AgentState
from langchain_core.messages import ToolMessage
from langchain_neo4j import AsyncNeo4jSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from typing_extensions import NotRequired
from api.utils.helpers import HTTPLoggedException

PERCEIVED_USER_GOAL = "perceived_user_goal"
APPROVED_USER_GOAL = "approved_user_goal"
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"
SEARCH_RESULTS = "search_results"
PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
NODE_CONSTRUCTION = "node_construction"
RELATIONSHIP_CONSTRUCTION = "relationship_construction"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"

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

agent_preset_names = {
    PERCEIVED_USER_GOAL: "perceived_user_goal",
    SUGGESTED_FILES:"suggested_files",
    PROPOSED_CONSTRUCTION_PLAN: "proposed_construction_plan"
}
        
LIST_THREADS_QUERY = """
MATCH (t:Thread {checkpoint_ns: ""})-[:HAS_CHECKPOINT]->(c:Checkpoint)
RETURN t.thread_id AS thread_id,
       count(c) AS checkpoints,
       toString(min(c.created_at)) AS created_at,
       toString(max(c.created_at)) AS updated_at
ORDER BY updated_at DESC
"""

FIND_THREAD_QUERY = """
MATCH (t:Thread {thread_id: $thread_id})-[:HAS_CHECKPOINT]->(c:Checkpoint)
RETURN t.thread_id AS thread_id,
       count(c) AS checkpoints,
       toString(min(c.created_at)) AS created_at,
       toString(max(c.created_at)) AS updated_at
"""
# AsyncNeo4jSaver.adelete_thread doesn't delete the thread branches: they would be left orphaned
DELETE_THREAD_BRANCHES_QUERY = """
MATCH (:Thread {thread_id: $thread_id})-[:HAS_BRANCH]->(b:Branch)
DETACH DELETE b
"""