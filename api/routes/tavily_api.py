from api.models.dto_schemas import TavilyAgentRequest
from api.utils.helpers import HTTPLoggedException
from api.utils.statics import default_llm_model
from api.mappers.tavily_mapper import toSearchTool, toExtractTool
from fastapi import APIRouter
from loguru import logger
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_tavily import TavilyExtract
import os

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
    })
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][-1]}")
    return response["messages"][-1].content

@router.post("/search")
async def service_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_search = TavilySearch(max_results=5, topic="general") if req.search_config is None else toSearchTool(req.search_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else default_llm_model,
        tools=[tavily_search],
        system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information and use it to answer the user." if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    })
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content

@router.post("/extract")
async def service_check(req: TavilyAgentRequest) -> str:
    # Initialize the Tavily Search tool
    tavily_extract = TavilyExtract(extract_depth="basic") if req.extract_config is None else toExtractTool(req.extract_config)

    # Initialize the agent with the search tool
    agent = create_agent(
        model=f"ollama:{req.model}" if req.model is not None else default_llm_model,
        tools=[tavily_extract],
        system_prompt="You are a helpful research assistant. Use the extract tool to gather information from the provided urls." if req.sysmsg is None else req.sysmsg
    )

    # Use the agent
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": req.input}]
    })
    logger.warning(f"-- AGENT RESPONSE --> {response["messages"][0:]}")
    return response["messages"][-1].content