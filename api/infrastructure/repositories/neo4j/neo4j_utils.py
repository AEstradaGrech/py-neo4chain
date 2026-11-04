# api/infrastructure/db/neo4j_drivers.py
from pathlib import Path

from loguru import logger
from neo4j import AsyncDriver
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.data_loader import MarkdownLoader
from neo4j_graphrag.components.types import TextChunk
import re
from collections.abc import Iterator
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from api.infrastructure.llm.agent_utils import ollama_embeddings
from api.infrastructure.llm.schemas.schemas import OllamaEmbeddingsConfig
from api.infrastructure.llm.statics import default_llm_model, default_embedding_model
from api.infrastructure.repositories.neo4j.statics import default_neo4jdb
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

from api.utils.helpers import validated_file_path

_kg_driver: AsyncDriver | None = None

def set_kg_driver(driver: AsyncDriver | None) -> None:
    global _kg_driver
    _kg_driver = driver

def kg_driver() -> AsyncDriver:
    if _kg_driver is None:
        raise RuntimeError("KG driver not initialized: it's created in the FastAPI lifespan")
    return _kg_driver

def set_kg_sync_driver(driver: GraphDatabase | None) -> None:
    global _kg_sync_driver
    _kg_sync_driver = driver

def kg_sync_driver() -> GraphDatabase:
    if _kg_sync_driver is None:
        raise RuntimeError("Sync KG driver not initialized: it's created in the FastAPI lifespan")
    return _kg_sync_driver

class RegexTextSplitter(TextSplitter):
    """Split text using regex matched delimiters."""
    def __init__(self, pattern: str):
        self.pattern = pattern

    def iter_chunks(self, text: str) -> Iterator[TextChunk]:
        for i, part in enumerate(re.split(self.pattern, text)):
            yield TextChunk(text=part, index=i)


class TitledMarkdownLoader(MarkdownLoader):
    def get_document_metadata(self, text: str, metadata: dict[str, str] | None = None) -> dict[str, str] | None:
        match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        return {**(metadata or {}), "title": match.group(1) if match else "Untitled"}

def simple_kg_pipeline(extraction_prompt:str, entity_schema, llm:str = None, embedding_model:str = None, db_name:str = None) -> SimpleKGPipeline:
    return SimpleKGPipeline(
        llm=OllamaLLM(model_name=default_llm_model if llm is None else llm),
        driver=kg_sync_driver(),                       # ¡síncrono! (ver abajo)
        embedder=OllamaEmbeddings(model=default_embedding_model if embedding_model is None else embedding_model),#ollama_embeddings(model=embeddings_config.model, dimensions=embeddings_config.dimensions, num_ctx=embeddings_config.num_ctx),
        from_file=True,                              # 'from_pdf' está deprecado
        file_loader=TitledMarkdownLoader(),          # 'pdf_loader' está deprecado
        text_splitter=RegexTextSplitter(r"\r?\n(?=## )"),
        schema=entity_schema,
        prompt_template=extraction_prompt,
        neo4j_database=default_neo4jdb if db_name is None else db_name,
    )

"""
entity_schema = {
    "node_types": schema_node_types,
    "relationship_types": schema_relationship_types,
    "patterns": schema_patterns,
    "additional_node_types": False, # True would be less strict, allowing unknown node types
}

# approved entities list can be used directly 
schema_node_types = approved_entities
    > schema_node_types:  ['Product', 'Issue', 'Feature', 'Location']

# the keys from approved fact types dictionary can be used for relationship types
schema_relationship_types = [key.upper() for key in approved_fact_types.keys()]
    > schema_relationship_types:  ['HAS_ISSUE', 'INCLUDES_FEATURE', 'USED_IN_LOCATION']

# rewrite the fact types into a list of tuples
schema_patterns = [
    [ fact['subject_label'], fact['predicate_label'].upper(), fact['object_label'] ]
    for fact in approved_fact_types.values()
]
    > schema_patterns: [['Product', 'HAS_ISSUE', 'Issue'], ['Product', 'INCLUDES_FEATURE', 'Feature'], ['Product', 'USED_IN_LOCATION', 'Location']]

"""

def file_context(file_path:str, num_lines=5) -> str:
    """Helper function to extract the first few lines of a file

    Args:
        file_path (str): Path to the file
        num_lines (int, optional): Number of lines to extract. Defaults to 5.

    Returns:
        str: First few lines of the file
    """
    with open(file_path, 'r') as f:
        lines = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    return "\n".join(lines)

def contextualize_er_extraction_prompt(context:str) -> str:
    """Creates a prompt with pre-amble file content for context during entity+relationship extraction.
    The context is concatenated into the string, which later will be used as a template
    for values like {schema} and {text}.
    """
    general_instructions = """
    You are a top-tier algorithm designed for extracting
    information in structured formats to build a knowledge graph.

    Extract the entities (nodes) and specify their type from the following text.
    Also extract the relationships between these nodes.

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
    "relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

    Use only the following node and relationship types (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Make sure you adhere to the following rules to produce valid JSON objects:
    - Do not return any additional information other than the JSON in it.
    - Omit any backticks around the JSON - simply output the JSON on its own.
    - The JSON object must not wrapped into a list - it is its own JSON object.
    - Property names must be enclosed in double quotes
    """

    context_placeholder = f"""
    Consider the following context to help identify entities and relationships:
    <context>
    {context}  
    </context>"""
    
    input_placeholder = """
    Input text:

    {text}
    """

    return general_instructions + "\n" + context_placeholder + "\n" + input_placeholder

def make_kg_builder(file_path:str, entity_schema, llm:str = None, embedding_model:str = None, db_name:str = None) -> SimpleKGPipeline:
    """Builds a KG builder for a given file, which is used to contextualize the chunking and entity extraction."""
    context = file_context(file_path)
    contextualized_prompt = contextualize_er_extraction_prompt(context)
    return simple_kg_pipeline(
        extraction_prompt=contextualized_prompt,
        entity_schema=entity_schema,
        llm=llm,
        embedding_model=embedding_model,
        db_name=db_name)

async def run_kg_builder(file_path:str, entity_schema, llm:str = None, embedding_model:str = None, db_name:str = None) -> PipelineResult:
    """Builds a KG builder for a given file, which is used to contextualize the chunking and entity extraction."""
    path = validated_file_path(file_path=file_path)
    context = file_context(path)
    contextualized_prompt = contextualize_er_extraction_prompt(context)
    pipeline =simple_kg_pipeline(
        extraction_prompt=contextualized_prompt,
        entity_schema=entity_schema,
        llm=llm,
        embedding_model=embedding_model,
        db_name=db_name)
    logger.warning(f"-- running KG pipeline for file: {file_path}")
    return await pipeline.run_async(file_path=path)

def normalize_key(label:str, key:str) -> str:
    """Normalizes a a property key for a given label.

    Keys are normalized by:
    - lowercase the key
    - remove any leading/trailing whitespace
    - remove label prefix from key
    - replace internal whitespace with "_"

    for example: 
        - "Product_name" -> "name"
        - "product name" -> "name"
        - "price" -> "price

    Args:
        label (str): The label to normalize keys for
        keys (List[str]): The list of keys to normalize

    Returns:
        List[str]: The normalized list of keys
    """
    lowercase_key = key.lower()
    unprefixed_key = re.sub(f"^{label.lower()}[_ ]*", "", lowercase_key)
    normalized_key = re.sub(" ", "_", unprefixed_key)
    return normalized_key