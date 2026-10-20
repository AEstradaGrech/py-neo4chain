from fastapi import APIRouter
from loguru import logger
from deepagents import create_deep_agent
from langchain.agents import create_agent
router = APIRouter(prefix="/tests")
import os

@router.get("/check") 
async def check()-> str:
    logger.warning("-- CHECK --")
    return "OK"

@router.get("/ollama-check") 
async def check()-> str:
    logger.warning("-- OLLAMA CHECK --")
    check_model = "ollama:gpt-oss:20b"
    instructions = """You are a helpful assistant. Greet the user using the 'greet_user' tool and answer using only the content of the tool output"""

    d_agent = create_deep_agent(
        model=check_model,
        tools=[greet_user],
        system_prompt=instructions,
    )
    d_result = await d_agent.ainvoke(
            {"messages": [{"role": "user", "content": "Hi"}]}
    )
    logger.warning("DEEP AGENT RESULT")
    print(d_result["messages"][-1].content)
    logger.warning("AGENT CHECK")
    agent = create_agent(
        model=check_model,
        tools=[greet_user],
        system_prompt=instructions,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Hi"}]}
    )
    logger.warning("AGENT RESULT")
    print(result["messages"][-1].content)
    return result["messages"][-1].content

def greet_user(city: str) -> str:
    """Get greet for user."""
    logger.warning("-- GREET TOOL CALL --")
    return f"AHOY!"