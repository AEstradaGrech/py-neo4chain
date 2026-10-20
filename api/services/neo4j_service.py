import os
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from api.utils.statics import default_neo4jdb, CYPHER_GENERATION_TEMPLATE, EXPANDED_CYPHER_GENERATION_TEMPLATE
from api.utils.helpers import ollama
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jService:
    _db_name:str
    def __init__(self, db_name:str = None):
        self._db_name = db_name if db_name is not None else default_neo4jdb

    def get_schema(self, db_name:str = None):
        graph = self.get_graph(db_name=db_name)
        logger.warning(f"-- DB SCHEMA >>\n\n{graph.schema}")
        return graph.schema
    
    def get_graph(self, db_name:str = None) -> Neo4jGraph:
        graph = Neo4jGraph(
            url=os.environ["NEO_URI"],
            username=os.environ["NEO_USERNAME"],
            password=os.environ["NEO_PASS"],
            database=self._db_name if db_name is None else db_name
        )
        logger.warning(f"-- graph for db: {self._db_name if db_name is None else db_name} --")
        graph.refresh_schema()
        return graph

    async def graph_qa_from_ollama(self, query:str, db_name:str = None) -> str:
        graph = self.get_graph()
        
        llm = ollama()
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], 
            template=self.get_qa_cypher_generator_msg()
        )
        cypher_msg = CYPHER_GENERATION_TEMPLATE.replace("{schema}", self.get_schema(db_name=db_name)).replace("{question}", query)
        chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, cypher_prompt=cypher_msg, allow_dangerous_requests=True, verbose=True)
        response = await chain.ainvoke({"query": query})
        logger.warning(f"-- NEO4J CHAIN RESPONSE:\n\n {response}")
        response["result"]



    def get_qa_cypher_generator_msg(self, expanded:bool = False) -> str:
        return CYPHER_GENERATION_TEMPLATE if expanded is False else EXPANDED_CYPHER_GENERATION_TEMPLATE