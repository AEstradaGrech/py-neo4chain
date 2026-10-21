from typing import Any, Dict
from api.models.dto_schemas import TavilyAgentRequest
from api.utils.helpers import HTTPLoggedException, tracing_config
from api.utils.statics import default_llm_model, tavily_research_poll_secs, tavily_research_timeout_secs
from api.mappers.tavily_mapper import toCrawlTool, toMapTool, toResearchTool, toSearchTool, toExtractTool
from fastapi import APIRouter
from loguru import logger
from langchain.agents import create_agent
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyResearch, TavilySearch, TavilyGetResearch
from langchain_tavily import TavilyExtract
import asyncio
import os
import time

router = APIRouter(prefix="/tavily")

@router.get("/check")
async def service_check() -> str:
    # Initialize the Tavily Search tool
    tavily_search = TavilySearch(max_results=5, topic="general")

    # Initialize the agent with the search tool
    agent = create_agent(
        model="ollama:gpt-oss:20b",
        tools=[tavily_search],
        system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information and use it to answer the user."
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the most popular sport in the world? Include only Wikipedia sources."}]
    }, config=tracing_config("tavily-check", ["tavily", "search"]))
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][-1]}")
    return response["messages"][-1].content

@router.post("/search")
async def search_agent_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_search = TavilySearch(max_results=5, topic="general") if req.search_config is None else toSearchTool(req.search_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else f"ollama:{default_llm_model}",
        tools=[tavily_search],
        system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information and use it to answer the user." if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    }, config=tracing_config("tavily-search", ["tavily", "search"]))
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content

@router.post("/extract")
async def extract_agent_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_extract = TavilyExtract(extract_depth="basic") if req.extract_config is None else toExtractTool(req.extract_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else f"ollama:{default_llm_model}",
        tools=[tavily_extract],
        system_prompt="You are a helpful research assistant. ALWAYS use the 'tavily_extract' tool to gather information from the provided urls." if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    }, config=tracing_config("tavily-extract", ["tavily", "extract"]))
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content

@router.post("/map")
async def map_agent_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_map = TavilyMap(extract_depth="basic") if req.map_config is None else toMapTool(req.map_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else f"ollama:{default_llm_model}",
        tools=[tavily_map],
        system_prompt="You are a helpful research assistant. Use the map tool to get a list of all the mentioned urls present in the content of the provided urls." if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    }, config=tracing_config("tavily-map", ["tavily", "map"]))
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content

@router.post("/crawl")
async def crawl_agent_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_crawl = TavilyCrawl(extract_depth="basic") if req.crawl_config is None else toCrawlTool(req.crawl_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else f"ollama:{default_llm_model}",
        tools=[tavily_crawl],
        system_prompt="You are a helpful research assistant. Use the crawl tool to extract information about the provided url." if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    }, config=tracing_config("tavily-crawl", ["tavily", "crawl"]))
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content

@router.post("/research")
async def research_agent_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_research = TavilyResearch(stream_results=False) if req.research_config is None else toResearchTool(req.research_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else f"ollama:{default_llm_model}",
        tools=[tavily_research],
        system_prompt="You are a helpful research assistant. Use the research tool to gather information and generate a research about the input task" if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    }, config=tracing_config("tavily-research", ["tavily", "research"]))
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content

@router.post("/research/direct")
async def research_direct_check(req: TavilyAgentRequest) -> Dict[str, Any]:
    # No agent here: a non-streamed research task only returns a request_id, so it is created it and poll until it's done
    tavily_research = TavilyResearch() if req.research_config is None else toResearchTool(req.research_config)
    tavily_get_research = TavilyGetResearch()

    task = await tavily_research.ainvoke({"input": req.input, "stream": False})
    if not isinstance(task, dict) or "error" in task:
        raise HTTPLoggedException(status_code=502, detail=f"Tavily research task creation failed: {task}")
    logger.warning(f"-- RESEARCH TASK CREATED --> {task}")

    deadline = time.monotonic() + tavily_research_timeout_secs
    while True:
        result = await tavily_get_research.ainvoke({"request_id": task["request_id"]})
        if not isinstance(result, dict) or "error" in result:
            raise HTTPLoggedException(status_code=502, detail=f"Tavily get research failed: {result}")

        status = result.get("status")
        if status == "completed":
            logger.warning(f"-- RESEARCH COMPLETED --> {task["request_id"]}")
            return result
        if status == "failed":
            raise HTTPLoggedException(status_code=502, detail=f"Tavily research task failed: {result}")
        if time.monotonic() > deadline:
            raise HTTPLoggedException(status_code=504, detail=f"Tavily research task {task["request_id"]} still '{status}' after {tavily_research_timeout_secs}s")

        logger.warning(f"-- RESEARCH STATUS --> {status}")
        await asyncio.sleep(tavily_research_poll_secs)