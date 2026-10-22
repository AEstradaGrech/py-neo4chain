
from typing import Any, Dict, List
from langchain.agents import AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from typing_extensions import NotRequired

class KGState(AgentState):
    """Agent state shared between tool calls.
    Pass it as create_agent(state_schema=KGState): Command updates on keys not declared here are silently dropped"""
    approved_user_goal: NotRequired[Dict[str, Any]]
    all_available_files: NotRequired[List[str]]
    suggested_files: NotRequired[List[str]]
    approved_files: NotRequired[List[str]]

def tool_success(key: str, result: Any) -> Dict[str, Any]:
    return {"status": "success", key: result}

def tool_error(message: str) -> Dict[str, Any]:
    return {"status": "error", "error_message": message}

def update_state(tool_runtime: ToolRuntime, key: str, value: Any) -> Command:
    """Writes key=value in the agent state and answers the tool call.
    Assigning tool_runtime.state[key] directly is lost: the state only changes through what a tool returns"""
    return Command(update={
        key: value,
        # a tool returning a Command must also return the ToolMessage that closes its tool call
        "messages": [ToolMessage(content=json.dumps(tool_success(key, value)), tool_call_id=tool_runtime.tool_call_id)],
    })