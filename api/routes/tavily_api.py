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