
from typing import Any

from fastapi import APIRouter
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from api.mappers.neo4j_mapper import schemaToDto
from api.models.dto_schemas import Neo4jSchema, TextFileVectorSearch
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

@router.post("/file/similarity-search")
async def similarity_search(req: TextFileVectorSearch) -> str:
    svc = Neo4jService(db_name="sotu")
    chunks = svc.split_text_file(req.textfile_path, chunk_size=req.chunk_size if req.chunk_size is not None else 1000, overlap=req.chunk_overlap if req.chunk_overlap is not None else 20)
    logger.warning(f"-- TOTAL CHUNKS: {len(chunks)} --")
    store = svc.get_store_from_docs(docs=chunks)
    search = store.similarity_search_with_score(req.query, k=2)
    logger.warning(f"-- NEO SEARCH TOP 2 --> {search}")
    retriever = store.as_retriever()
    results = retriever.invoke(req.query)
    logger.warning(f"-- NEO RESULTS --> {results}")
    return results