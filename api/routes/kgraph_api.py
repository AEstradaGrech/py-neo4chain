
from pathlib import Path
from typing import Any, Dict, List
import uuid
from fastapi import APIRouter, Request
from api.infrastructure.llm.agents_factory import AgentsFactory

from api.infrastructure.llm.agent_utils import StepKey, agent_config, get_agent_state, get_thread_values, nudge_message, step_response
from api.infrastructure.llm.models.ner_agent import NER_NUDGE
from api.infrastructure.llm.models.facts_agent import FACTS_NUDGE
from api.infrastructure.llm.models.schema_proposal_agent import SCHEMA_PROPOSAL_NUDGE
from api.infrastructure.llm.statics import ALL_AVAILABLE_FILES, APPROVED_FILES, APPROVED_USER_GOAL, PERCEIVED_USER_GOAL, PROPOSED_CONSTRUCTION_PLAN, SUGGESTED_FILES, APPROVED_CONSTRUCTION_PLAN, APPROVED_ENTITIES, APPROVED_FACTS, DELETE_THREAD_BRANCHES_QUERY, FIND_THREAD_QUERY, INSPECTED_NER_FILES, LIST_THREADS_QUERY, PROPOSED_ENTITIES, PROPOSED_FACTS
from api.infrastructure.repositories.neo4j.kgraph_repository import KGraphRepository
from api.infrastructure.repositories.neo4j.neo4j_utils import make_kg_builder, run_kg_builder
from api.models.kgraph_schemas import KGAgentRequestDto, KGApproveRequestDto, KGPipelineRequestDto, KGSchemaRequestDto, KGStepResponseDto
from api.utils.helpers import HTTPLoggedException
from api.infrastructure.llm.agent_utils import kgraph_intent_tracing
from api.infrastructure.llm.statics import facts_max_nudges, ner_max_nudges, schema_proposal_max_nudges
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
    await validated_request_thread(req.thread_id, request)
    factory = AgentsFactory()
    #check KEY exists
    agent = factory.kg_agent_by_step_key(step_key=req.state_key, model=req.model, checkpointer=request.app.state.neo4j_saver)
    if agent is None:
        raise HTTPLoggedException(status_code=400, detail=f"No agent has been found for key: {req.state_key}")
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
    factory = AgentsFactory()
    agent = factory.kg_agent_by_step_key(
        step_key=StepKey.PERCEIVED_USER_GOAL, 
        model=req.model, 
        checkpointer=request.app.state.neo4j_saver, 
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        num_predict=req.max_tokens,
        retries=req.retries, 
        retry_delay=req.retry_delay)
    config = agent_config(thread_id, kgraph_intent_tracing())
    messages = [{"role": "user", "content": req.input}]
    response = await agent.ainvoke({"messages" : messages},config=config)
    print(response)
    if PERCEIVED_USER_GOAL not in response: 
        #the LLM may have asked a clarifying question. offer some more details
        return KGStepResponseDto(
            thread_id=thread_id,  # req.thread_id is None on a new thread: the client needs the generated one to answer
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
    await validated_request_thread(req.thread_id, request)
    await check_thread_values(thread_id=req.thread_id, values=[APPROVED_USER_GOAL], request=request)
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    messages = [{"role": "user", "content": req.input}]
    factory = AgentsFactory()
    agent = factory.kg_agent_by_step_key(
        step_key=StepKey.SUGGESTED_FILES, 
        model=req.model, 
        checkpointer=request.app.state.neo4j_saver,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        num_predict=req.max_tokens,
        retries=req.retries,
        retry_delay=req.retry_delay)
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
    await validated_request_thread(req.thread_id, request)
    await check_thread_values(thread_id=req.thread_id, values=[APPROVED_USER_GOAL, APPROVED_FILES], request=request)
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    nudges = 0
    factory = AgentsFactory()
    if req.num_refinements is not None and req.num_refinements > 0:
        # sub agents without checkpointer: each one starts a fresh conversation, the loop graph persists the shared state
        # (goal, files and plan are read from the thread)
        schema_refiner = factory.kg_schema_refining_loop(
            checkpointer=request.app.state.neo4j_saver,
            max_iterations=req.num_refinements,
            retries=req.retries,
            retry_delay=req.retry_delay
        )
        response = await schema_refiner.ainvoke({"messages": []}, config=config)
    else:
        schema_agent = factory.kg_agent_by_step_key(
            step_key=StepKey.PROPOSED_CONSTRUCTION_PLAN,
            model=req.model, 
            checkpointer=request.app.state.neo4j_saver,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            num_predict=req.max_tokens, 
            retries=req.retries, 
            retry_delay=req.retry_delay)
        logger.warning(f"-- PROPOSE SCHEMA >> USER INPUT: {req.input} --")
        response = await schema_agent.ainvoke({"messages": [{"role": "user", "content": req.input}]}, config=config)
        # gpt-oss:20b sometimes stops after sampling the files: nudge it to go on in the same thread (it keeps its work)
        while not response.get(PROPOSED_CONSTRUCTION_PLAN) and nudges < schema_proposal_max_nudges:
            nudges += 1
            logger.warning(f"-- PROPOSE SCHEMA >> EMPTY PLAN, NUDGE {nudges}/{schema_proposal_max_nudges} --")
            response = await schema_agent.ainvoke({"messages": [{"role": "user", "content": SCHEMA_PROPOSAL_NUDGE}]}, config=config)
    if req.comfy:
        approval_agent = factory.kg_agent_by_step_key(
            step_key=StepKey.SCHEMA_REFINER, 
            model=req.model, 
            checkpointer=request.app.state.neo4j_saver,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            num_predict=req.max_tokens, 
            retries=req.retries, 
            retry_delay=req.retry_delay)
        response = await approval_agent.ainvoke({"messages": [{"role": "user", "content": "Ok, approved"}]}, config=config)
    
    return step_response(thread_id=req.thread_id, name=StepKey.PROPOSED_CONSTRUCTION_PLAN, response=response, approved=req.comfy)


@router.post("/refined-schema")
async def refined_schema(req: KGSchemaRequestDto, request:Request) -> Any:
    await validated_request_thread(req.thread_id, request)
    await check_thread_values(thread_id=req.thread_id, values=[APPROVED_USER_GOAL, APPROVED_FILES], request=request) 
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    factory = AgentsFactory()
    schema_refiner = factory.kg_agent_by_step_key(
        step_key=StepKey.SCHEMA_REFINER, 
        model=req.model,
        checkpointer=request.app.state.neo4j_saver, 
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        num_predict=req.max_tokens,
        retries=req.retries, 
        retry_delay=req.retry_delay)
    messages = [{"role": "user", "content": req.input}]
    response = await schema_refiner.ainvoke({"messages": messages}, config=config)
    if req.comfy:
        response = await schema_refiner.ainvoke({"messages": [{"role": "user", "content": "Ok, approved"}]}, config=config)
    return step_response(StepKey.PROPOSED_CONSTRUCTION_PLAN, thread_id=req.thread_id, response=response, approved=True if APPROVED_CONSTRUCTION_PLAN in response else False)
    
@router.post("/ner-schema")
async def ner_schema(req: KGAgentRequestDto, request:Request) -> Any:
    await validated_request_thread(req.thread_id, request)
    await check_thread_values(thread_id=req.thread_id, request=request, values=[APPROVED_USER_GOAL, APPROVED_FILES, APPROVED_CONSTRUCTION_PLAN])
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    factory = AgentsFactory()
    agent = factory.kg_agent_by_step_key(
        step_key=StepKey.PROPOSED_ENTITIES, 
        model=req.model,
        checkpointer=request.app.state.neo4j_saver, 
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        num_predict=req.max_tokens,
        retries=req.retries,
        retry_delay=req.retry_delay)
    messages = [{"role": "user", "content": req.input}]
    response = await agent.ainvoke({"messages": messages}, config=config)
    # gpt-oss:20b sometimes presents the entity types but doesn't save them: nudge it to finish in the same thread
    # (nudge messages keep the current request context, see current_request_only)
    nudges = 0
    while not response.get(PROPOSED_ENTITIES) and nudges < ner_max_nudges:
        nudges += 1
        logger.warning(f"-- NER SCHEMA >> NO ENTITY TYPES SAVED, NUDGE {nudges}/{ner_max_nudges} --")
        response = await agent.ainvoke({"messages": [nudge_message(NER_NUDGE)]}, config=config)
    if req.comfy and response.get(PROPOSED_ENTITIES):
        response = await approve_in_state(agent, config, PROPOSED_ENTITIES, APPROVED_ENTITIES)
    return step_response(StepKey.PROPOSED_ENTITIES, response=response, thread_id=req.thread_id, approved=True if APPROVED_ENTITIES in response else False)

@router.post("/facts-schema")
async def facts_schema(req: KGAgentRequestDto, request:Request) -> Any:
    await validated_request_thread(req.thread_id, request)
    # facts relate the approved entity types, found in the text files inspected by the NER step
    await check_thread_values(thread_id=req.thread_id, request=request,
                              values=[APPROVED_USER_GOAL, APPROVED_CONSTRUCTION_PLAN, APPROVED_ENTITIES, INSPECTED_NER_FILES])
    config = agent_config(req.thread_id, kgraph_intent_tracing())
    factory = AgentsFactory()
    agent = factory.kg_agent_by_step_key(
        step_key=StepKey.PROPOSED_FACTS,
        model=req.model,
        checkpointer=request.app.state.neo4j_saver,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        num_predict=req.max_tokens,
        retries=req.retries,
        retry_delay=req.retry_delay)
    messages = [{"role": "user", "content": req.input}]
    response = await agent.ainvoke({"messages": messages}, config=config)
    # gpt-oss:20b sometimes presents the fact types but doesn't record them: nudge it to finish in the same thread
    # (nudge messages keep the current request context, see current_request_only)
    nudges = 0
    while not response.get(PROPOSED_FACTS) and nudges < facts_max_nudges:
        nudges += 1
        logger.warning(f"-- FACTS SCHEMA >> NO FACT TYPES RECORDED, NUDGE {nudges}/{facts_max_nudges} --")
        response = await agent.ainvoke({"messages": [nudge_message(FACTS_NUDGE)]}, config=config)
    if req.comfy and response.get(PROPOSED_FACTS):
        response = await approve_in_state(agent, config, PROPOSED_FACTS, APPROVED_FACTS)
    return step_response(StepKey.PROPOSED_FACTS, response=response, thread_id=req.thread_id, approved=True if APPROVED_FACTS in response else False)

@router.get("/insert-construction-plan/{thread_id}")
async def fill_kg_db(thread_id:str, request:Request):
    repo = KGraphRepository()
    await validated_request_thread(thread_id=thread_id, request=request)
    state = await get_thread_state(thread_id=thread_id, request=request)
    if state is None:
        raise HTTPLoggedException(status_code=400, detail=f"No agent state has been found for thread_id: {thread_id}")
    await check_thread_values(thread_id=thread_id, values=[APPROVED_CONSTRUCTION_PLAN,APPROVED_ENTITIES, APPROVED_ENTITIES], request=request)
    response = await repo.construct_domain_graph(state.get(APPROVED_CONSTRUCTION_PLAN))
    return response
    
@router.post("/build-kg")
async def build_kg(req: KGPipelineRequestDto, request: Request) -> Any:
    repo = KGraphRepository()
    await validated_request_thread(thread_id=req.thread_id, request=request)
    state = await get_thread_state(thread_id=req.thread_id, request=request)
    if state is None:
        raise HTTPLoggedException(status_code=400, detail=f"No agent state has been found for thread_id: {req.thread_id}")
    await check_thread_values(thread_id=req.thread_id, values=[APPROVED_CONSTRUCTION_PLAN, APPROVED_ENTITIES, APPROVED_ENTITIES, INSPECTED_NER_FILES], request=request)
    # TODO: check inserted
    inserted_construction_plan = await repo.construct_domain_graph(state.get(APPROVED_CONSTRUCTION_PLAN))
    if "nodes" not in inserted_construction_plan or "relationships" not in inserted_construction_plan:
        raise HTTPLoggedException(status_code=500, detail=f"An error has occured while inserting the construction plan for thread id: {req.thread_id}")    
    results = {}
    schema_node_types = state.get(APPROVED_ENTITIES)
    approved_facts = state.get(APPROVED_FACTS)
    schema_relationship_types = [key.upper() for key in approved_facts.keys()]
    schema_patterns = [[ fact['subject_label'], fact['predicate_label'].upper(), fact['object_label']] for fact in approved_facts.values()]
    entity_schema = {
        "node_types": schema_node_types,
        "relationship_types": schema_relationship_types,
        "patterns": schema_patterns,
        "additional_node_types": False, # True would be less strict, allowing unknown node types
    }
    logger.warning(f">> ENTITY SCHEMA:\n\n{entity_schema}")
    for product_review in state.get(INSPECTED_NER_FILES):
        results[product_review] = await run_kg_builder(file_path=product_review, entity_schema=entity_schema, llm=req.llm_model, embedding_model=req.embedding_model, db_name=req.db_name)
    return results


@router.delete("clear-data")
async def clear_kgraph_data(full_delete:bool):
    repo = KGraphRepository()
    return await repo.clear_neo4j_data(is_full_delete=full_delete)
    
async def approve_in_state(agent: Any, config: Dict[str, Any], proposed_key: str, approved_key: str) -> Dict[str, Any]:
    """comfy approval in code: copies the proposal to its approved key and returns the updated state.
    Agents with current_request_only take an 'Ok, approved' message as a new request: they don't see their
    proposal, so instead of approving it they start the task again (and overwrite it)"""
    state = (await agent.aget_state(config)).values
    await agent.aupdate_state(config, {approved_key: state[proposed_key]})
    return (await agent.aget_state(config)).values

async def validated_request_thread(thread_id:str, request:Request) -> Any:
    if thread_id is None or thread_id == "":
        raise HTTPLoggedException(status_code=400, detail="No thread id has been found in the request.")
    thread = await get_thread(thread_id=thread_id, request=request)
    if thread is None or thread["thread_id"] != thread_id:
        raise HTTPLoggedException(status_code=404, detail="Request thread id not found in DB")
    return thread

async def check_thread_values(thread_id:str, values: List[str], request:Request) -> bool:
    thread_values = await get_thread_values(request, thread_id)
    for value in values:
        if value not in thread_values:
            raise HTTPLoggedException(status_code=400, detail=f"The thread state has no {value}.")
    