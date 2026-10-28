from typing import Any
from api.infrastructure.llm.agent_utils import KGState
from api.infrastructure.llm.tools.kgraph_tools import get_approved_files, get_approved_user_goal, get_proposed_construction_plan, sample_file, search_file
from api.utils.helpers import agent_chat_model
from langchain.agents import create_agent

DEFAULT_AGENT_NAME = "schema_critic_agent"
DEFAULT_DESCRIPTION = "Criticizes the proposed schema for relevance to the user goal and approved files."

critic_agent_role_and_goal = """
# TASK: 
    You are an expert at knowledge graph modeling with property graphs. 
    Criticize the proposed schema for relevance to the user goal and approved files.
"""

critic_agent_hints = """
# INSTRUCTIONS:
    Criticize the proposed schema for relevance and correctness:
    - Are unique identifiers actually unique? Use the 'search_file' tool to validate. Composite identifier are not acceptable.
    - Could any nodes be relationships instead? Double-check that unique identifiers are unique and not references to other nodes. Use the 'search_file' tool to validate
    - Can you manually trace through the source data to find the necessary information for anwering a hypothetical question?
    - Is every node in the schema connected? What relationships could be missing? Every node should connect to at least one other node.
    - Are hierarchical container relationships missing? 
    - Are any relationships redundant? A relationship between two nodes is redundant if it is semantically equivalent to or the inverse of another relationship between those two nodes.
"""

critic_agent_chain_of_thought_directions = """
    Prepare for the task:
    - get the user goal using the 'get_approved_user_goal' tool
    - get the list of approved files using the 'get_approved_files' tool
    - get the construction plan using the 'get_proposed_construction_plan' tool
    - use the 'sample_file' and 'search_file' tools to validate the schema design

# RULES:
    Think carefully, using tools to perform actions and reconsidering your actions when a tool returns an error:
    1. Analyze each construction rule in the proposed construction plan.
    2. Use tools to validate the construction rules for relevance and correctness.
    3. If the schema looks good, respond with a one word reply: 'valid'.
    4. If the schema has problems, respond with 'retry' and provide feedback as a concise bullet list of problems.
"""
def schema_critic_agent(checkpointer: Any, model:str = None, name:str = None, description:str = None):
    if name is not None and name.lower() == "default":
        name = DEFAULT_AGENT_NAME
    if description is not None and description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    return create_agent(
        model=agent_chat_model(model),
        name=name,
        state_schema=KGState,
        checkpointer=checkpointer,
        tools=[
            get_approved_user_goal, 
            get_approved_files,
            get_proposed_construction_plan,
            sample_file, 
            search_file
        ],
        system_prompt=uset_intent_instruction(description)
    )

def uset_intent_instruction(description: None):
    instruction = f"""
{critic_agent_role_and_goal}
{critic_agent_hints}
{critic_agent_chain_of_thought_directions}
"""
    return instruction if description is None else f"# AGENT DESCRIPTION: {description}\n\n{instruction}"