
from typing import Any, Dict, List
import uuid
from fastapi import APIRouter, Request
from api.infrastructure.llm.agents_factory import AgentsFactory
from api.infrastructure.llm.models.file_suggester_agent import file_suggester_agent
from api.infrastructure.llm.agent_utils import DELETE_THREAD_BRANCHES_QUERY, FIND_THREAD_QUERY, LIST_THREADS_QUERY, agent_config, get_agent_state, get_thread_values, step_response
from api.infrastructure.llm.models.schema_critic_agent import schema_critic_agent
from api.infrastructure.llm.models.schema_proposal_agent import SCHEMA_PROPOSAL_NUDGE, schema_proposal_agent
from api.infrastructure.llm.models.schema_refiner_agent import schema_refinement_loop
from api.infrastructure.llm.agent_utils import ALL_AVAILABLE_FILES, APPROVED_FILES, APPROVED_USER_GOAL, PERCEIVED_USER_GOAL, PROPOSED_CONSTRUCTION_PLAN, SUGGESTED_FILES
from api.models.kgraph_schemas import AgentRequestDto, KGAgentRequestDto, KGApproveRequestDto, KGSchemaRequestDto, KGStepResponseDto, MessageDto
from api.infrastructure.llm.models.user_intent_agent import user_intent_agent
from api.infrastructure.llm.models.file_suggester_agent import file_suggester_agent
from api.utils.helpers import HTTPLoggedException, kgraph_intent_tracing
from api.utils.statics import schema_proposal_max_nudges
from api.mappers.prompting_mapper import to_openai_messages
from loguru import logger

router = APIRouter(prefix="/dlai/knowledge-graph-construction")

@router.get("/threads")
async def get_threads(request: Request) -> List[Dict[str, Any]]:
    records, _, _ = await request.app.state.neo4j_driver.execute_query(LIST_THREADS_QUERY)
    return [r.data() for r in records]

@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, request:Request) -> Dict[str, Any] | None:
    driver = request.app.state.neo4j_driver
    thread, _, _ = await driver.execute_query(FIND_THREAD_QUERY, thread_id=thread_id)
    return thread[0].data() if len(thread) > 0 else None

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

@router.post("/approve-step")
async def approve_step(req: KGApproveRequestDto, request: Request) -> Any:
    await validated_request_thread(req, request)
    factory = AgentsFactory()
    #check KEY exists
    agent = factory.kgagent_by_step_key(step_key=req.state_key, model=req.model, checkpointer=request.app.state.neo4j_saver)
    state = await get_agent_state(agent, req.thread_id)
    if req.state_key not in state:
        raise HTTPLoggedException(status_code=400, detail="The step to approve is not present in the agent state")
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    messages = [{"role": "user", "content": req.input}]
    response = await agent.ainvoke({"messages" : messages},config=config)
    return response

@router.post("/user-intent")

async def user_intent(req: KGAgentRequestDto, request: Request) -> Any:
    thread_id = req.thread_id or str(uuid.uuid4())
    agent = user_intent_agent(model=req.model, description="default", checkpointer=request.app.state.neo4j_saver)
    config = agent_config(thread_id, kgraph_intent_tracing())
    messages = [{"role": "user", "content": req.input}]
    response = await agent.ainvoke({"messages" : messages},config=config)
    print(response)
    if PERCEIVED_USER_GOAL not in response: 
        #the LLM may have asked a clarifying question. offer some more details
        return KGStepResponseDto(
            thread_id=req.thread_id,
            state_key=PERCEIVED_USER_GOAL,
            status="needs_clarification",
            message=response["messages"][-1].content)     
    if req.comfy:
        messages = [{"role": "user", "content": "Approve that goal."}]
        response = await agent.ainvoke({"messages" : messages},config=config)
        return step_response(PERCEIVED_USER_GOAL,thread_id=thread_id,response=response, approved=True)
    return step_response(PERCEIVED_USER_GOAL,thread_id=thread_id,response=response, approved=False)

@router.post("/suggest-files")
async def files_suggestion(req: KGAgentRequestDto, request: Request) -> Any:
    await validated_request_thread(req, request)
    
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    messages = [{"role": "user", "content": req.input}]
    agent = file_suggester_agent(model=req.model, description="default", checkpointer=request.app.state.neo4j_saver)
    #thread_state = await get_agent_state(agent, thread_id=req.thread_id)
    response = await agent.ainvoke({"messages" : messages},config=config)
    if ALL_AVAILABLE_FILES not in response:
        return KGStepResponseDto(
            thread_id=req.thread_id,
            state_key=SUGGESTED_FILES,
            status="needs_clarification",
            message=response["messages"][-1].content)
    if SUGGESTED_FILES not in response:
        return KGStepResponseDto(
            thread_id=req.thread_id,
            state_key=SUGGESTED_FILES,
            status="needs_clarification",
            message=response["messages"][-1].content,
            output_state={
                ALL_AVAILABLE_FILES: response.get(ALL_AVAILABLE_FILES),
        })
    if req.comfy:
      messages = [{"role": "user", "content": "Ok, use the suggested files"}]
      response = await agent.ainvoke({"messages": messages}, config=config)
      return step_response(SUGGESTED_FILES,thread_id=req.thread_id,response=response, approved=True)
    return step_response(SUGGESTED_FILES,thread_id=req.thread_id,response=response, approved=False)

@router.post("/propose-schema")
async def propose_schema(req: KGSchemaRequestDto, request:Request) -> Any:
    await validated_request_thread(req, request)
    thread_values = await get_thread_values(request, req.thread_id)
    if APPROVED_USER_GOAL not in thread_values or APPROVED_FILES not in thread_values:
        raise HTTPLoggedException(status_code=409, detail="The thread has no approved user goal or approved files yet. Use /user-intent and /suggest-files first.")
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    nudges = 0

    if req.num_refinements is not None and req.num_refinements > 0:
        # sub agents without checkpointer: each one starts a fresh conversation, the loop graph persists the shared state
        # (goal, files and plan are read from the thread)
        schema_refiner = schema_refinement_loop(
            proposal_agent=schema_proposal_agent(model=req.model, name=req.name, description=req.description, checkpointer=None),
            critic_agent=schema_critic_agent(model=req.model, name="default", description="default", checkpointer=None),
            checkpointer=request.app.state.neo4j_saver,
            max_iterations=req.num_refinements
        )
        response = await schema_refiner.ainvoke({"messages": []}, config=config)
    else:
        schema_agent = schema_proposal_agent(model=req.model, name=req.name, description=req.description, checkpointer=request.app.state.neo4j_saver)
        logger.warning(f"-- PROPOSE SCHEMA >> USER INPUT: {req.input} --")
        response = await schema_agent.ainvoke({"messages": [{"role": "user", "content": req.input}]}, config=config)
        # gpt-oss:20b sometimes stops after sampling the files: nudge it to go on in the same thread (it keeps its work)
        while not response.get(PROPOSED_CONSTRUCTION_PLAN) and nudges < schema_proposal_max_nudges:
            nudges += 1
            logger.warning(f"-- PROPOSE SCHEMA >> EMPTY PLAN, NUDGE {nudges}/{schema_proposal_max_nudges} --")
            response = await schema_agent.ainvoke({"messages": [{"role": "user", "content": SCHEMA_PROPOSAL_NUDGE}]}, config=config)

    return KGStepResponseDto(
        thread_id=req.thread_id,
        state_key=PROPOSED_CONSTRUCTION_PLAN,
        status="graph_schema" if response.get(PROPOSED_CONSTRUCTION_PLAN) else "graph_construction_error",
        message=response["messages"][-1].content,
        output_state={
            PROPOSED_CONSTRUCTION_PLAN: response.get(PROPOSED_CONSTRUCTION_PLAN)
        })


async def validated_request_thread(req: AgentRequestDto, request:Request) -> Any:
    if req.thread_id is None or req.thread_id == "":
        raise HTTPLoggedException(status_code=400, detail="No thread id has been found in the request.")
    thread = await get_thread(thread_id=req.thread_id, request=request)
    if thread is None or thread["thread_id"] != req.thread_id:
        raise HTTPLoggedException(status_code=404, detail="Request thread id not found in DB")
    return thread