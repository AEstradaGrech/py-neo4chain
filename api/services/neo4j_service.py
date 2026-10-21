import os
from typing import List
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from neo4j_graphrag.types import SearchType
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from api.utils.statics import default_neo4jdb, CYPHER_GENERATION_TEMPLATE, EXPANDED_CYPHER_GENERATION_TEMPLATE
from api.utils.helpers import ollama, ollama_embeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class Neo4jService:
    _db_name:str
    def __init__(self, db_name:str = None):
        self._db_name = db_name if db_name is not None else default_neo4jdb

    def get_schema(self, db_name:str = None):
        graph = self.get_graph(db_name=db_name)
        return graph.schema
    
    def get_graph(self, db_name:str = None) -> Neo4jGraph:
        graph = Neo4jGraph(
            url=os.environ["NEO_URI"],
            username=os.environ["NEO_USERNAME"],
            password=os.environ["NEO_PASS"],
            database=self._db_name if db_name is None else db_name
        )
        graph.refresh_schema()
        return graph

    def get_store_from_docs(self, docs:List[Document], hybrid:bool = False, embedder:OllamaEmbeddings = None) -> Neo4jVector:
        return Neo4jVector.from_documents(
            documents=docs,
            embedding=embedder if embedder is not None else ollama_embeddings(),
            search_type=SearchType.HYBRID if hybrid else SearchType.VECTOR,
            url=os.environ["NEO_URI"], 
            username=os.environ["NEO_USERNAME"], 
            password=os.environ["NEO_PASS"]
        )

    def get_store_from_index(self, index_name:str, index_keyword:str, hybrid:bool = False, embedder:OllamaEmbeddings = None) -> Neo4jVector:
        return Neo4jVector.from_existing_index(
            index_name=index_name,
            keyword_index_name=index_keyword,
            embedding=embedder if embedder is not None else ollama_embeddings(),
            search_type=SearchType.HYBRID if hybrid else SearchType.VECTOR,
            url=os.environ["NEO_URI"], 
            username=os.environ["NEO_USERNAME"], 
            password=os.environ["NEO_PASS"]
        )

    def get_store_from_graph(self, node_label:str, node_properties: List[str], hybrid:bool = False, embedder:OllamaEmbeddings = None) -> Neo4jVector:
        return Neo4jVector.from_existing_graph(
            embedding=embedder if embedder is not None else ollama_embeddings(),
            node_label=node_label,
            text_node_properties=node_properties,
            search_type=SearchType.HYBRID if hybrid else SearchType.VECTOR,
            url=os.environ["NEO_URI"], 
            username=os.environ["NEO_USERNAME"], 
            password=os.environ["NEO_PASS"]
        )

    def show_indexes(self, db_name:str = None):
        graph = self.get_graph(db_name=db_name)
        return graph.query("""SHOW VECTOR INDEXES""")

    def neo4j_index_vector_search(graph: Neo4jGraph, index_name:str, question:str, top_k = 5):
        vector_search_query = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $question_embedding) yield node, score
            RETURN score, node.text AS text
        """
        question_embedding = ollama_embeddings.embed_query(question)
        return graph.query(vector_search_query, 
            params={
            'question_embedding': question_embedding, 
            'index_name':index_name, 
            'top_k': top_k})
    
    def split_text_file(self, path:str, chunk_size:int = 1000, overlap:int = 20):
        loader = TextLoader(f"{path}.txt",encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return text_splitter.split_documents(documents)
    
    async def graph_qa_from_ollama(self, query:str, db_name:str = None) -> str:
        graph = self.get_graph()
        llm = ollama()
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], 
            template=self.get_qa_cypher_generator_msg()
        )
        chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, cypher_prompt=CYPHER_GENERATION_PROMPT, allow_dangerous_requests=True, verbose=True)
        response = await chain.ainvoke({"query": query})
        return response["result"]
        

    def get_qa_cypher_generator_msg(self, expanded:bool = False) -> str:
        return CYPHER_GENERATION_TEMPLATE if expanded is False else EXPANDED_CYPHER_GENERATION_TEMPLATE