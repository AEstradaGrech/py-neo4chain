
from typing import Any

from api.infrastructure.llm.agent_utils import KGState, current_request_only, ollama_retry_middleware
from api.infrastructure.llm.tools.kgraph_tools import approve_proposed_entities, get_approved_user_goal, get_proposed_entities, get_well_known_types, list_folder_files, list_folders, sample_file, set_proposed_entities
from api.infrastructure.llm.agent_utils import agent_chat_model
from api.infrastructure.llm.statics import default_agent_numctx
from langchain.agents import create_agent

DEFAULT_AGENT_NAME="ner_schema_agent" 
DEFAULT_DESCRIPTION="Proposes the kind of named entities that could be extracted from text files."
# sent when the agent finishes without saving entity types (gpt-oss:20b sometimes presents them but skips the tool)
NER_NUDGE = """Continue with the task: save the entity types you proposed using the 'set_proposed_entities' tool,
then present them to the user using the 'get_proposed_entities' tool."""

ner_agent_role_and_goal = """
# TASK:
  You are a top-tier algorithm designed for analyzing text files and proposing
  the kind of named entities that could be extracted which would be relevant 
  for a user's goal.
  """

ner_agent_hints = """
# INSTRUCTIONS:
  Entities are people, places, things and qualities, but not quantities. 
  Your goal is to propose a list of the type of entities, not the actual instances
  of entities.

  There are two general approaches to identifying types of entities:
  - well-known entities: these closely correlate with approved node labels in an existing graph schema
  - discovered entities: these may not exist in the graph schema, but appear consistently in the source text

  Design rules for well-known entities:
  - always use existing well-known entity types. For example, if there is a well-known type "Person", and people appear in the text, then propose "Person" as the type of entity.
  - prefer reusing existing entity types rather than creating new ones
  
  Design rules for discovered entities:
  - discovered entities are consistently mentioned in the text and are highly relevant to the user's goal
  - always look for entities that would provide more depth or breadth to the existing graph
  - for example, if the user goal is to represent social communities and the graph has "Person" nodes, look through the text to discover entities that are relevant like "Hobby" or "Event"
  - avoid quantitative types that may be better represented as a property on an existing entity or relationship.
  - for example, do not propose "Age" as a type of entity. That is better represented as an additional property "age" on a "Person".
  - prefer a few general entity types over many specific ones: if several candidate types are variants of the same thing
    (for example "AssemblyIssue", "HardwareIssue", "MaterialIssue"), propose one general type ("Issue") and keep the
    variant as a property (an "Issue" with a "category" property)
  - include the people who wrote the text when they are relevant to the user's goal (for example, the reviewer of a product review)
"""

ner_agent_chain_of_thought_directions = """
# RULES:
  Follow these steps in order, using the tools. Don't answer the user until step 6 is done:
  1. Use 'get_approved_user_goal' to get the user goal.
  2. Use 'list_folders' to find the folder the user wants to use as data source to inspect.
  3. Use 'list_folder_files' to get the files to analyze, and 'get_well_known_types' to get the node labels already in the graph.
  4. Use 'sample_file' on EVERY inspected file. Take note of the people, places, things and qualities mentioned.
  5. Decide the entity types: reuse the well-known types that appear in the text, and add the discovered types
     that appear consistently and support the user goal, following the design rules for discovered entities
     (for example, one general type for the problems customers complain about).
  6. Save them with 'set_proposed_entities' (a list of type names), then use 'get_proposed_entities' and present them
     to the user for approval.
  7. Only if the user approves, use 'approve_proposed_entities'.
"""

def ner_agent(checkpointer: Any, model:str = None, name:str = None, description:str = None,temperature:float = 0.7, top_p:float = 0.9, top_k:int=40, num_predict:int = -2, ctx_len: int = default_agent_numctx, retries:int = 0, retry_delay:float =0.5):
    if name is not None and name.lower() == "default":
        name = DEFAULT_AGENT_NAME
    if description is not None and description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    return create_agent(
        model=agent_chat_model(model, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len),
        name=name,
        state_schema=KGState,
        checkpointer=checkpointer,
        # current_request_only: the shared thread history (previous pipeline steps) overflows the context window
        middleware=[current_request_only] + ([ollama_retry_middleware(retries, retry_delay)] if retries > 0 else []),
        tools=[
            get_approved_user_goal, 
            list_folders,
            list_folder_files, 
            sample_file,
            get_well_known_types,
            get_proposed_entities,
            set_proposed_entities,
            approve_proposed_entities
        ],
        system_prompt=ner_instruction(description)
    )

def ner_instruction(description: None):
    instruction = f"""
{ner_agent_role_and_goal}
{ner_agent_hints}
{ner_agent_chain_of_thought_directions}
"""
    return instruction if description is None else f"# AGENT DESCRIPTION: {description}\n{instruction.strip()}"
