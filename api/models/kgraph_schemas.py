

from typing import List, Optional

from pydantic import BaseModel, Field

class MessageDto(BaseModel):
    role: str = Field(description="Chat role. 'system' | 'user' | 'assistant' | 'tool_call'")
    message:str = Field(description="Message content")

class AgentRequestDto(BaseModel):
    input:str = Field(description="User input")
    thread_id: Optional[str] = Field(description="Thread ID. Leave empty to create a new thread id", default=None)
    model:Optional[str] = Field(description="model to use with provider in this format -> <provider>:<model>. Empty defaults to ollama:gpt-oss:20b.", default=None)
    sysmsg:Optional[str] = Field(description="Instruction message for the agent", default=None)
    

