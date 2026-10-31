"""LangGraph version of the ADK schema refinement loop from 
    https://learn.deeplearning.ai/courses/agentic-knowledge-graph-construction/lesson/jjx7m/schema-proposal-for-structured-data:

    LoopAgent(max_iterations=2, sub_agents=[schema_proposal_agent, schema_critic_agent, CheckStatusAndEscalate])

    START -> start_loop -> schema_proposal_agent -> schema_critic_agent --(feedback 'valid' or max iterations)--> END
                                  ^                                   |
                                  +--------------(retry)--------------+
"""
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import Command, CompiledStateGraph
from langchain.tools import tool, ToolRuntime
from loguru import logger
from typing_extensions import NotRequired
from langchain.agents import create_agent
from api.infrastructure.llm.agent_utils import APPROVED_FILES, APPROVED_USER_GOAL, FEEDBACK, PROPOSED_CONSTRUCTION_PLAN, KGState, dict_delta, ollama_retry_middleware, state_update, tool_success
from api.infrastructure.llm.models.schema_critic_agent import schema_critic_agent
from api.infrastructure.llm.models.schema_proposal_agent import SCHEMA_PROPOSAL_NUDGE, schema_proposal_agent
from api.infrastructure.llm.tools.kgraph_tools import  approve_proposed_construction_plan, get_proposed_construction_plan
from api.utils.helpers import agent_chat_model
from api.utils.statics import schema_proposal_max_nudges

DEFAULT_AGENT_NAME = "schema_refiner_agent"
DEFAULT_DESCRIPTION = "Generates an enhanced version of the the proposed knowledge graph construction schema"
DEFAULT_MAX_ITERATIONS = 2
VALID_FEEDBACK = "valid"  # the critic replies 'valid' (one word) or 'retry' + a bullet list of problems
REFINEMENT_ITERATIONS = "refinement_iterations"
PROPOSAL_NODE = "schema_proposal_agent"
CRITIC_NODE = "schema_critic_agent"
# state keys the sub agents need: they read goal, files and plan through their tools
SHARED_KEYS = (APPROVED_USER_GOAL, APPROVED_FILES, PROPOSED_CONSTRUCTION_PLAN, FEEDBACK)

class SchemaRefinementState(KGState):
    refinement_iterations: NotRequired[int]

def sub_agent_input(state: SchemaRefinementState, task: str) -> Dict[str, Any]:
    """Each sub agent starts a fresh conversation with only the shared state keys,
    so the critic doesn't get the proposal agent tool calls (and the context doesn't grow every iteration)"""
    return {**{key: state[key] for key in SHARED_KEYS if key in state}, "messages": [HumanMessage(task)]}

def is_valid(feedback: str) -> bool:
    return feedback.strip().strip(".'\"").lower() == VALID_FEEDBACK

def schema_refinement_loop(proposal_agent: CompiledStateGraph, critic_agent: CompiledStateGraph,
                           checkpointer: Any = None, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> CompiledStateGraph:
    """Generator/critic loop. Sub agents are created without checkpointer: the loop graph persists the shared state"""

    async def start_loop(state: SchemaRefinementState) -> Dict[str, Any]:
        # the iteration counter is persisted in the thread: reset it on every run of the loop
        return {REFINEMENT_ITERATIONS: 0}

    async def propose(state: SchemaRefinementState) -> Dict[str, Any]:
        result = await proposal_agent.ainvoke(sub_agent_input(state, "Propose a knowledge graph schema for the approved files."))
        # the plan has a merge_dicts reducer: send only the changes, so the rules removed by the agent are removed here too
        plan_changes = dict_delta(state.get(PROPOSED_CONSTRUCTION_PLAN, {}), result.get(PROPOSED_CONSTRUCTION_PLAN, {}))
        return {PROPOSED_CONSTRUCTION_PLAN: plan_changes, "messages": [result["messages"][-1]]}

    async def critique(state: SchemaRefinementState) -> Dict[str, Any]:
        result = await critic_agent.ainvoke(sub_agent_input(state, "Criticize the proposed construction plan."))
        return {
            FEEDBACK: result["messages"][-1].content,  # ADK: output_key="feedback"
            REFINEMENT_ITERATIONS: state.get(REFINEMENT_ITERATIONS, 0) + 1,
            "messages": [result["messages"][-1]],
        }

    def check_status_and_escalate(state: SchemaRefinementState) -> Literal["schema_proposal_agent", "__end__"]:
        """ADK: CheckStatusAndEscalate (escalate when feedback is 'valid') + LoopAgent(max_iterations)"""
        if is_valid(state.get(FEEDBACK, VALID_FEEDBACK)) or state.get(REFINEMENT_ITERATIONS, 0) >= max_iterations:
            return END
        return PROPOSAL_NODE

    builder = StateGraph(SchemaRefinementState)
    builder.add_node("start_loop", start_loop)
    builder.add_node(PROPOSAL_NODE, propose)
    builder.add_node(CRITIC_NODE, critique)
    
    builder.add_edge(START, "start_loop")
    builder.add_edge("start_loop", PROPOSAL_NODE)
    builder.add_edge(PROPOSAL_NODE, CRITIC_NODE)
    builder.add_conditional_edges(CRITIC_NODE, check_status_and_escalate)
    return builder.compile(checkpointer=checkpointer, name="schema_refinement_loop")

schema_proposal_coordinator_instruction = """
# TASK:
    You coordinate the schema proposal process. You don't design the schema yourself: the 'refine_proposed_plan' tool does it.

# RULES:
    1. Call the 'refine_proposed_plan' tool first to create (or improve) the proposed construction plan. It takes no arguments.
    2. Use the 'get_proposed_construction_plan' tool to get the plan and present it to the user for approval.
    3. If the user disapproves, call 'refine_proposed_plan' again and present the new plan.
    4. If the user approves, use the 'approve_proposed_construction_plan' tool to record the approval.
"""
def schema_refiner_agent(checkpointer: Any, model:str = None, name:str = None, description:str = None,retries:int = 0, retry_delay:float =0.5):
    if name is not None and name.lower() == "default":
        name = DEFAULT_AGENT_NAME
    if description is not None and description.lower() == "default":
        description = DEFAULT_DESCRIPTION
    logger.warning(f"-- REFINER AGENT INSTRUCTION >> \n\n{refiner_agent_instruction(description)}")
    return create_agent(
        model=agent_chat_model(model),
        name=name,
        state_schema=KGState,
        checkpointer=checkpointer,
        middleware=[ollama_retry_middleware(retries, retry_delay)] if retries > 0 else [],
        tools=[refine_proposed_plan, get_proposed_construction_plan, approve_proposed_construction_plan],
        system_prompt=refiner_agent_instruction(description)
    )

def refiner_agent_instruction(description: None):
    instruction = f"{schema_proposal_coordinator_instruction}"
    return instruction if description is None else f"# AGENT DESCRIPTION: {description}\n\n{instruction}"

@tool
async def refine_proposed_plan(tool_runtime: ToolRuntime) -> Command:
    """Proposes a knowledge graph schema (construction plan) for the approved files and refines it with a critic.
    Creates the plan if there is none yet, or improves the current one. Takes no arguments: it reads the
    approved user goal, approved files and current plan from the agent state.

    Returns:
            dict: A dictionary containing metadata about the content.
                    Includes a 'status' key ('success' or 'error').
                    If 'success', includes a 'proposed_construction_plan' key with the refined construction plan.
    """
    logger.warning("-- REFINE PROPOSED PLAN TOOL CALL --")
    # 2 iterations: propose -> critique -> propose again applying the feedback (with 1 the critique is never applied)
    schema_refiner = schema_refinement_loop(
            proposal_agent=schema_proposal_agent(name="default", description="default", checkpointer=None),
            critic_agent=schema_critic_agent(name="default", description="default", checkpointer=None),
            max_iterations=DEFAULT_MAX_ITERATIONS
        )
    nudges = 0
    # the agent state is already here (loaded by the agent checkpointer): no thread_id or checkpointer needed.
    # No config either: the tracing callbacks are inherited from the agent run
    result = await schema_refiner.ainvoke(
        {**{key: tool_runtime.state[key] for key in SHARED_KEYS if key in tool_runtime.state}, "messages": []})
    while not result.get(PROPOSED_CONSTRUCTION_PLAN) and nudges < schema_proposal_max_nudges:
        nudges += 1
        logger.warning(f"-- PROPOSE SCHEMA >> EMPTY PLAN, NUDGE {nudges}/{schema_proposal_max_nudges} --")
        result = await schema_refiner.ainvoke({"messages": [{"role": "user", "content": SCHEMA_PROPOSAL_NUDGE}]})
    # merge_dicts reducer: send only the changes, so the rules removed by the loop are removed in the agent state too
    plan_changes = dict_delta(tool_runtime.state.get(PROPOSED_CONSTRUCTION_PLAN, {}), result.get(PROPOSED_CONSTRUCTION_PLAN, {}))
    return state_update(tool_runtime,
                        {PROPOSED_CONSTRUCTION_PLAN: plan_changes, FEEDBACK: result.get(FEEDBACK)},
                        tool_success(PROPOSED_CONSTRUCTION_PLAN, result.get(PROPOSED_CONSTRUCTION_PLAN, {})))