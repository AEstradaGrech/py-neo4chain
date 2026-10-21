
from typing import Any, List

from fastapi import APIRouter
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.documents import Document
from api.mappers.neo4j_mapper import schemaToDto
from api.models.dto_schemas import Neo4jSchema, TextFileVectorSearch
from api.services.neo4j_service import Neo4jService


router = APIRouter(prefix="/neo4j")

@router.get("/show-indexes")
async def show_db_indexes() -> List[str]:
    svc = Neo4jService()
    return svc.show_indexes()

@router.get("/schema")
async def db_schema() -> Neo4jSchema:
    svc = Neo4jService()
    return schemaToDto(svc.get_schema())

@router.get("/check/seed-data")
async def check() -> Any:
    svc = Neo4jService()
    logger.warning(f"-- SCHEMA -- \n\n{svc.get_schema()}")
    graph = svc.get_graph()
    query = """
MATCH (m:Movie)
RETURN count(m) as numberOfMovies
    """
    query_res =graph.query(query=query.strip())
    logger.warning(f"-- SIMPLE GRAPH QUERY\n\nQuery: {query}\n\nResult: {query_res}")
    query = """
MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie) 
RETURN actor.name, movie.title LIMIT 10
    """
    query_res =graph.query(query=query.strip())
    logger.warning(f"-- SIMPLE GRAPH QUERY\n\nQuery: {query}\n\nResult: {query_res}")
    query = """
MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie) 
RETURN actor.name, movie.title LIMIT 10
        """
    query_res =graph.query(query=query.strip())
    logger.warning(f"-- SIMPLE GRAPH QUERY\n\nQuery: {query}\n\nResult: {query_res}")
    query = """
MATCH (actor:Person {name:"Kirsten Dunst"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coactors) 
RETURN actor.name, m.title, coactors.name
    """
    query_res =graph.query(query=query.strip())
    logger.warning(f"-- SIMPLE GRAPH QUERY\n\nQuery: {query}\n\nResult: {query_res}")
    response = await svc.graph_qa_from_ollama(query="What was the cast of Jumanji?")
    return response

@router.post("/check/file/similarity-search")
async def similarity_search(req: TextFileVectorSearch) -> List[Document]:
    svc = Neo4jService(db_name="sotu")
    chunks = svc.split_text_file(req.textfile_path, chunk_size=req.chunk_size if req.chunk_size is not None else 1000, overlap=req.chunk_overlap if req.chunk_overlap is not None else 20)
    store = svc.get_store_from_docs(docs=chunks)
    # individual API
    #search = store.similarity_search_with_score(req.query, k=req.k)
    retriever = store.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": req.k if req.k is not None else 4, "score_threshold":0.0 if req.min_score is None else req.min_score})
    results = retriever.invoke(req.query)
    return results