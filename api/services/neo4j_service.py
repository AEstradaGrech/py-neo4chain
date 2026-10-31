import os
from typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from neo4j_graphrag.types import SearchType
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from api.utils.statics import default_neo4jdb, CYPHER_GENERATION_TEMPLATE, EXPANDED_CYPHER_GENERATION_TEMPLATE
from api.utils.helpers import HTTPLoggedException, ollama, ollama_embeddings
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

    def neo4j_is_ready(self):
        return self.get_graph().send_query("RETURN 'Neo4j is Ready!' as message")

    def drop_neo4j_indexes(self):
        """Drops and constraints and indexes present on the neo4j graph database

        Returns:
            Success or an error.
        """
        # remove all constraints
        list_constraints = self.get_graph().send_query(
            """SHOW CONSTRAINTS YIELD name"""
        )
        if (list_constraints == "error"):
            return list_constraints
        constraint_names = [row["name"] for row in list_constraints["query_result"]]
        for constraint_name in constraint_names:
            dropped_constraint = self.get_graph().send_query("""DROP CONSTRAINT $constraint_name""", {"constraint_name": constraint_name})
            if (dropped_constraint["status"] == "error"):
                return dropped_constraint

        # remove all indexes
        list_indexes = self.get_graph().send_query(
            """SHOW INDEXES YIELD name"""
        )
        if (list_indexes == "error"):
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while listing the DB indexes: {list_indexes}")
        index_names = [row["name"] for row in list_indexes["query_result"]]
        for index_name in index_names:
            dropped_index = self.get_graph().send_query("""DROP INDEX $index_name""", {"index_name": index_name})
            if (dropped_index["status"] == "error"):
                raise HTTPLoggedException(status_code=500, detail=f"An error has occured while dropping the DB index: {dropped_index}")
            
        return list_indexes
    
    def clear_neo4j_data(self) -> bool:
        """Clears all data from the neo4j graph database.

        Use with caution! Confirm with the user
        that they know this will completely reset the database.

        Returns:
            True if Success 
        """
        # First, remove all nodes and relationships in batches
        data_removed = self.get_graph().send_query("""MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS""")
        if (data_removed["status"] == "error") :
            raise HTTPLoggedException(status_code=500, detail=f"An error has occured while clearing the DB data: {data_removed}")

        return True

    def get_apoc_procedure_names(self):
        """List all APOC procedure names.
        APOC (Awesome Procedures on Cypher) is a library of procedures and functions that extends the capabilities of Neo4j.

        Returns:
            Success with a list of APOC procedure names or an error.
        """
        cypher = "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN name"

        result = self.get_graph().send_query(cypher)
        
        if result["status"] == "error":
            return result
        apoc_procedure_names = [row["name"] for row in result["query_result"]]

        if len(apoc_procedure_names) == 0:
            raise HTTPLoggedException(status_code=404,detail="APOC procedures not found. Please ensure APOC is installed in your Neo4j database.")
        
        return apoc_procedure_names

    def get_apoc_version(self) -> Dict[str, Any]:
        """Get the version of APOC installed in the Neo4j database.

        Returns:
            Success with the APOC version or an error.
        """
        cypher = "RETURN apoc.version() AS apoc_version"

        result = self.get_graph().send_query(cypher)
        
        if result["status"] == "error":
            raise HTTPLoggedException(status_code=404,detail="An error has occured while checking the apoc version")
        
        return result["query_result"][0]["apoc_version"]
        
    def get_neo4j_version(self) -> Dict[str, Any]:
        """Get the version and edition of the Neo4j database.
        
        """
        cypher = "CALL dbms.components() yield name, versions, edition unwind versions as version return name, version, edition"

        result = self.get_graph().send_query(cypher)

        return result
        

    def create_uniqueness_constraint(self, label: str,unique_property_key: str):
        """Creates a uniqueness constraint for a node label and property key.
        A uniqueness constraint ensures that no two nodes with the same label and property key have the same value.
        This improves the performance and integrity of data import and later queries.

        Args:
            label: The label of the node to create a constraint for.
            unique_property_key: The property key that should have a unique value.

        Returns:
            A dictionary with a status key ('success' or 'error').
            On error, includes an 'error_message' key.
        """    
        # Use string formatting since Neo4j doesn't support parameterization of labels and property keys when creating a constraint
        constraint_name = f"{label}_{unique_property_key}_constraint"
        query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
        FOR (n:`{label}`)
        REQUIRE n.`{unique_property_key}` IS UNIQUE"""
        results = self.get_graph().send_query(query)
        return results
    
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