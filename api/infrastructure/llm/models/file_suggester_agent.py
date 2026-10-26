from typing import Any

from api.infrastructure.llm.models.agent_utils import KGState
from api.infrastructure.llm.tools.kgraph_tools import approve_suggested_files, get_approved_user_goal, get_suggested_files, list_available_files, sample_file, set_suggested_files
from api.utils.statics import default_llm_model
from langchain.agents import create_agent

DEFAULT_DESCRIPTION = "Helps the user select files to import."

file_suggestion_agent_instruction = """
# TASK:
You are a constructive critic AI reviewing a list of files. Your goal is to suggest relevant files
for constructing a knowledge graph.

# INSTRUCTIONS:
Review the file list for relevance to the kind of graph and description specified in the approved user goal. 

For any file that you're not sure about, use the 'sample_file' tool to get 
a better understanding of the file contents. 

Only consider structured data files like CSV or JSON.

Prepare for the task:
- use the 'get_approved_user_goal' tool to get the approved user goal

# RULES:
Think carefully, repeating these steps until finished:
1. list available files using the 'list_available_files' tool
2. evaluate the relevance of each file, then record the list of suggested files using the 'set_suggested_files' tool
3. use the 'get_suggested_files' tool to get the list of suggested files
4. ask the user to approve the set of suggested files
5. If the user has feedback, go back to step 1 with that feedback in mind
6. If approved, use the 'approve_suggested_files' tool to record the approval
"""

def file_suggester_agent(checkpointer: Any, model:str = None, description:str = None):
    if description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    return create_agent(
        model=f"ollama:{model}" if model is not None else f"ollama:{default_llm_model}",
        state_schema=KGState,
        checkpointer=checkpointer,
        tools=[
            get_approved_user_goal, 
            list_available_files, 
            sample_file, 
            set_suggested_files, 
            get_suggested_files,
            approve_suggested_files
        ],
        system_prompt=file_suggestion_agent_instruction if description is None else f"# AGENT DESCRIPTION: {description}\n\n{file_suggestion_agent_instruction}"
    )