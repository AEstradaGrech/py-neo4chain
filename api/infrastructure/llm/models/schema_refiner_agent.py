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
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired

from api.infrastructure.llm.models.agent_utils import KGState, dict_delta
from api.infrastructure.llm.tools.kgraph_tools import APPROVED_FILES, APPROVED_USER_GOAL, FEEDBACK, PROPOSED_CONSTRUCTION_PLAN

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
