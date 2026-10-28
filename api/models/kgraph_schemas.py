

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class MessageDto(BaseModel):
    role: str = Field(description="Chat role. 'system' | 'user' | 'assistant' | 'tool_call'")
    message:str = Field(description="Message content")

class AgentRequestDto(BaseModel):
    input:str = Field(description="User input")
    thread_id: Optional[str] = Field(description="Thread ID. Leave empty to create a new thread id", default=None)
    model:Optional[str] = Field(description="model to use with provider in this format -> <provider>:<model>. Empty defaults to ollama:gpt-oss:20b.", default=None)
    sysmsg:Optional[str] = Field(description="Instruction message for the agent", default=None)
    name:Optional[str] = Field(description="Agent Name. Pass 'default' for KGRAPH agents to use the default name", default=None)
    description:Optional[str] = Field(description="Agent Description (use it for agent purpose description). Pass 'default' for KGRAPH agents to use the default description", default=None)

class KGAgentRequestDto(AgentRequestDto):
    comfy: bool = Field(description="Flag to indicate whether to trust the agent generated state variables in case of tool success or wait for user approval", default=False)

class KGSchemaRequestDto(AgentRequestDto):
    num_refinements: Optional[int] = Field(description="Num of iterations for the schema_critic_agent. None or 0 to get the raw output of the schema_proposal agent", default=None)
    

class KGApproveRequestDto(BaseModel):
    input:Optional[str] = Field(description="User input", default="Ok, approved.")
    thread_id: Optional[str] = Field(description="Thread ID. Leave empty to create a new thread id", default=None)
    state_key: str = Field(description="KGState KEY to approve")

class KGStepResponseDto(BaseModel):
    thread_id: str = Field(description="Thread ID. Leave empty to create a new thread id")
    state_key: str = Field(description="KGState KEY of the runned step")
    status:str = Field(description="Tag with step status info")
    message: Any = Field(description="LLM message (if any)", default=None)
    output_state: dict[str, Any] = Field(description="Agent State to return", default={})