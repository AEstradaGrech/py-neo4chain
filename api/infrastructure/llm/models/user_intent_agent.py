
#"Helps the user ideate on a knowledge graph use case."
from typing import Any

from api.infrastructure.llm.models.agent_utils import KGState
from api.infrastructure.llm.tools.kgraph_tools import approve_perceived_user_goal, set_perceived_user_goal
from api.utils.statics import default_llm_model
from langchain.agents import create_agent

DEFAULT_DESCRIPTION = "Helps the user ideate on a knowledge graph use case."

agent_role_and_goal = """
# TASK:
You are an expert at knowledge graph use cases. 
Your primary goal is to help the user come up with a knowledge graph use case.
"""

agent_conversational_hints = """
# INSTRUCTIONS:
If the user is unsure what to do, make some suggestions based on classic use cases like:
- social network involving friends, family, or professional relationships
- logistics network with suppliers, customers, and partners
- recommendation system with customers, products, and purchase patterns
- fraud detection over multiple accounts with suspicious patterns of transactions
- pop-culture graphs with movies, books, or music
"""

agent_output_definition = """
A user goal has two components:
- kind_of_graph: at most 3 words describing the graph, for example "social network" or "USA freight logistics"
- description: a few sentences about the intention of the graph, for example "A dynamic routing and delivery system for cargo." or "Analysis of product dependencies and supplier alternatives."
"""

agent_chain_of_thought_directions = """
# RULES:
Think carefully and collaborate with the user:
1. Understand the user's goal, which is a kind_of_graph with description
2. Ask clarifying questions as needed
3. When you think you understand their goal, use the 'set_perceived_user_goal' tool to record your perception
4. Present the perceived user goal to the user for confirmation
5. If the user agrees, use the 'approve_perceived_user_goal' tool to approve the user goal. This will save the goal in state under the 'approved_user_goal' key.
"""

def user_intent_agent(checkpointer: Any, model:str = None, description:str = None):
    if description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    return create_agent(
        model=f"ollama:{model}" if model is not None else f"ollama:{default_llm_model}",
        state_schema=KGState,
        checkpointer=checkpointer,
        tools=[set_perceived_user_goal, approve_perceived_user_goal],
        system_prompt=uset_intent_instruction(description)
    )

def uset_intent_instruction(description: None):
    instruction = f"""
{agent_role_and_goal}
{agent_conversational_hints}
{agent_output_definition}
{agent_chain_of_thought_directions}
"""
    return instruction if description is None else f"# AGENT DESCRIPTION: {description}\n\n{instruction}"