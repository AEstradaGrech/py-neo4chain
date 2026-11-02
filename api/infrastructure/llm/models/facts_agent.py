from typing import Any

from api.infrastructure.llm.agent_utils import KGState, current_request_only, ollama_retry_middleware
from api.infrastructure.llm.tools.kgraph_tools import add_proposed_fact, approve_proposed_facts, get_approved_entities, get_approved_user_goal, get_inspected_files, get_proposed_facts, sample_file
from api.utils.helpers import agent_chat_model
from api.utils.statics import default_agent_numctx
from langchain.agents import create_agent

DEFAULT_AGENT_NAME = "fact_type_extraction_agent"
DEFAULT_DESCRIPTION = "Proposes the kind of relevant facts that could be extracted from text files."
# sent when the agent finishes without recording fact types (gpt-oss:20b sometimes presents them but skips the tool)
FACTS_NUDGE = """Continue with the task: record each fact type you proposed using the 'add_proposed_fact' tool,
then present them to the user using the 'get_proposed_facts' tool."""

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
# RULES:
  Follow these steps in order, using the tools. Don't answer the user until step 5 is done:
  1. Use 'get_approved_user_goal' to get the user goal, and 'get_approved_entities' to get the approved entity types.
  2. Use 'get_inspected_files' to get the text files to analyze.
  3. Use 'sample_file' on EVERY inspected file. Take note of how the approved entity types are related in the text.
  4. Decide the fact types (subject, predicate, object): subject and object must be approved entity types,
     and the predicate must appear in the text.
  5. Call 'add_proposed_fact' once for each fact type, then use 'get_proposed_facts' and present them to the user
     for approval, along with an explanation.
  6. Only if the user approves, use 'approve_proposed_facts'.
"""

def facts_agent(checkpointer: Any, model:str = None, name:str = None, description:str = None, temperature:float = 0.7, top_p:float = 0.9, top_k:int=40, num_predict:int = -2, ctx_len: int = default_agent_numctx, retries:int = 0, retry_delay:float =0.5):
    if name is not None and name.lower() == "default":
        name = DEFAULT_AGENT_NAME
    if description is not None and description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    return create_agent(
        model=agent_chat_model(model, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=num_predict, ctx_len=ctx_len),
        name=name,
        state_schema=KGState,
        checkpointer=checkpointer,
        middleware=[current_request_only] + ([ollama_retry_middleware(retries, retry_delay)] if retries > 0 else []),
        tools=[
            get_approved_user_goal,
            get_inspected_files,
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