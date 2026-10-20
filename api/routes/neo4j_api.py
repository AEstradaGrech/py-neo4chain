
from typing import Any

from fastapi import APIRouter
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from api.services.neo4j_service import Neo4jService


router = APIRouter(prefix="/neo4j")


@router.get("/check")
async def check() -> Any:
    svc = Neo4jService()
    graph = svc.get_graph()
    query = """
    MATCH (m:Movie)
    RETURN count(m) as numberOfMovies
    """
    query_res =graph.query(query=query)
    logger.warning(f"-- SIMPLE GRAPH QUERY --> {query_res}")
    response = await svc.graph_qa_from_ollama(query="What was the cast of The Matrix?")
    return response