from typing import Any

from api.infrastructure.llm.agent_utils import KGState, ollama_retry_middleware
from api.infrastructure.llm.tools.kgraph_tools import add_proposed_fact, approve_proposed_facts, get_approved_entities, get_approved_files, get_approved_user_goal, get_proposed_facts, sample_file
from api.utils.helpers import agent_chat_model
from langchain.agents import create_agent

DEFAULT_AGENT_NAME = "fact_type_extraction_agent"
DEFAULT_DESCRIPTION = "Proposes the kind of relevant facts that could be extracted from text files."

fact_agent_role_and_goal = """
# TASK:
  You are a top-tier algorithm designed for analyzing text files and proposing
  the type of facts that could be extracted from text that would be relevant 
  for a user's goal. 
"""

fact_agent_hints = """
# INSTRUCTIONS:

  Do not propose specific individual facts, but instead propose the general type 
  of facts that would be relevant for the user's goal. 
  For example, do not propose "ABK likes coffee" but the general type of fact "Person likes Beverage".
  
  Facts are triplets of (subject, predicate, object) where the subject and object are
  approved entity types, and the proposed predicate provides information about
  how they are related. For example, a fact type could be (Person, likes, Beverage).

  Design rules for facts:
  - only use approved entity types as subjects or objects. Do not propose new types of entities
  - the proposed predicate should describe the relationship between the approved subject and object
  - the predicate should optimize for information that is relevant to the user's goal
  - the predicate must appear in the source text. Do not guess.
  - use the 'add_proposed_fact' tool to record each proposed fact type
"""

fact_agent_chain_of_thought_directions = """
    Prepare for the task:
    - use the 'get_approved_user_goal' tool to get the user goal
    - use the 'get_approved_files' tool to get the list of approved files
    - use the 'get_approved_entities' tool to get the list of approved entity types

    Think step by step:
    1. Use the 'get_approved_user_goal' tool to get the user goal
    2. Sample some of the approved files using the 'sample_file' tool to understand the content
    3. Consider how subjects and objects are related in the text
    4. Call the 'add_proposed_fact' tool for each type of fact you propose
    5. Use the 'get_proposed_facts' tool to retrieve all the proposed facts
    6. Present the proposed types of facts to the user, along with an explanation
"""

def facts_agent(checkpointer: Any, model:str = None, name:str = None, description:str = None,retries:int = 0, retry_delay:float =0.5):
    if name is not None and name.lower() == "default":
        name = DEFAULT_AGENT_NAME
    if description is not None and description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    return create_agent(
        model=agent_chat_model(model),
        name=name,
        state_schema=KGState,
        checkpointer=checkpointer,
        middleware=[ollama_retry_middleware(retries, retry_delay)] if retries > 0 else [],
        tools=[
            get_approved_user_goal, 
            get_approved_files, 
            get_approved_entities,
            sample_file,
            add_proposed_fact,
            get_proposed_facts,
            approve_proposed_facts
        ],
        system_prompt=fact_instruction(description)
    )

def fact_instruction(description: None):
    instruction = f"""
{fact_agent_role_and_goal}
{fact_agent_hints}
{fact_agent_chain_of_thought_directions}
"""
    return instruction if description is None else f"# AGENT DESCRIPTION: {description}\n{instruction.strip()}"