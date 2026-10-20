
from typing import Any

from fastapi import APIRouter
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from api.mappers.neo4j_mapper import schemaToDto
from api.models.dto_schemas import Neo4jSchema
from api.services.neo4j_service import Neo4jService


router = APIRouter(prefix="/neo4j")

@router.get("/schema-string")
async def db_schema_string() -> str:
    svc = Neo4jService()
    return svc.get_schema()

@router.get("/schema")
async def db_schema() -> Neo4jSchema:
    svc = Neo4jService()
    return schemaToDto(svc.get_schema())

@router.get("/check")
async def check() -> Any:
    svc = Neo4jService()
    
    schema = svc.get_schema()
    logger.warning(f"-- SCHEMA -- \n\n{schema}")
    graph = svc.get_graph()
    query = """
    MATCH (m:Movie)
    RETURN count(m) as numberOfMovies
    """
    query_res =graph.query(query=query)
    logger.warning(f"-- SIMPLE GRAPH QUERY --> {query_res}")
    response = await svc.graph_qa_from_ollama(query="What was the cast of The Matrix?")
    return response