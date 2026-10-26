
from typing import Any, Dict, List
import uuid
from fastapi import APIRouter, Request
from api.infrastructure.llm.models.agent_utils import DELETE_THREAD_BRANCHES_QUERY, LIST_THREADS_QUERY, agent_config, get_agent_state, get_thread_values
from api.infrastructure.llm.tools.kgraph_tools import APPROVED_USER_GOAL, PERCEIVED_USER_GOAL
from api.models.kgraph_schemas import AgentRequestDto, MessageDto
from api.infrastructure.llm.models.user_intent_agent import user_intent_agent
from api.utils.helpers import HTTPLoggedException, kgraph_intent_tracing
from api.mappers.prompting_mapper import to_openai_messages
from loguru import logger

router = APIRouter(prefix="/dlai/knowledge-graph-construction")

@router.get("/threads")
async def get_threads(request: Request) -> List[Dict[str, Any]]:
    records, _, _ = await request.app.state.neo4j_driver.execute_query(LIST_THREADS_QUERY)
    return [r.data() for r in records]

@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, request: Request) -> Dict[str, Any]:
    driver = request.app.state.neo4j_driver
    records, _, _ = await driver.execute_query(
        "MATCH (t:Thread {thread_id: $thread_id}) RETURN count(t) > 0 AS found", thread_id=thread_id)
    if not records[0]["found"]:
        raise HTTPLoggedException(status_code=404, detail=f"Thread '{thread_id}' not found")

    await driver.execute_query(DELETE_THREAD_BRANCHES_QUERY, thread_id=thread_id)
    await request.app.state.neo4j_saver.adelete_thread(thread_id)
    return {"thread_id": thread_id, "deleted": True}

@router.get("/get-state/{thread_id}")
async def get_thread_state(thread_id: str, request: Request):
    values = await get_thread_values(request, thread_id)
    values.pop("messages", None)  # they have their own endpoint
    return values

@router.get("/messages/{thread_id}")
async def get_thread_messages(thread_id: str, request: Request):
    values = await get_thread_values(request, thread_id)
    return to_openai_messages(values.get("messages", []))

# TEST INPUT: "I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, which can support root-cause analysis."
# CLARIFY: "I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, which can support root-cause analysis."
@router.post("/user-intent")
async def user_intent(req: AgentRequestDto, request: Request) -> Any:
    thread_id = req.thread_id or str(uuid.uuid4())
    agent = user_intent_agent(model=req.model, description="default", checkpointer=request.app.state.neo4j_saver)
    config = agent_config(thread_id, kgraph_intent_tracing())
    messages = [{"role": "user", "content": req.input}]
    response = await agent.ainvoke({"messages" : messages},config=config)
    print(response)
    if PERCEIVED_USER_GOAL not in response: 
        #the LLM may have asked a clarifying question. offer some more details
        return response     
    # ELSE: Optimistically presume approval.
    messages = [{"role": "user", "content": "Approve that goal."}]
    response = await agent.ainvoke({"messages" : messages},config=config)

    return {
        "thread_id": thread_id,
        "status": "approved" if APPROVED_USER_GOAL in response else "needs_clarification",
        "message": response["messages"][-1].content,
        PERCEIVED_USER_GOAL: response.get(PERCEIVED_USER_GOAL),
        APPROVED_USER_GOAL: response.get(APPROVED_USER_GOAL),
    }