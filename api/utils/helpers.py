import os
from typing import Any, Dict, List
from fastapi import HTTPException
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langfuse.langchain import CallbackHandler
from loguru import logger

from api.utils.statics import default_llm_model, default_embedding_model, default_embedding_dimensions, default_embedding_numctx

def ollama(model:str = None, temp:float = 0.7, max_tokens:int = 600, ctx_len:int = 4096, verbose:bool = False) -> ChatOllama:
    logger.warning(f"-- OLLAMA INSTANCE FOR: {default_llm_model if model is None else model}")
    return ChatOllama(model=default_llm_model if model is None else model, temperature=temp, num_predict=max_tokens, num_ctx=ctx_len, verbose=verbose)

def ollama_embeddings(dimensions:int = None, model:str = None, num_ctx:int = None) -> OllamaEmbeddings:
    return  OllamaEmbeddings(
        model=default_embedding_model if model is None else model,
        dimensions=default_embedding_dimensions if dimensions is None else dimensions,
        num_ctx=default_embedding_numctx if num_ctx is None else num_ctx 
    )
def tracing_config(trace_name: str, tags: List[str] = None) -> Dict[str, Any]:
    """RunnableConfig that sends the run to Langfuse (keys from .env). Empty config (no tracing) if LANGFUSE_PUBLIC_KEY isn't set"""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return {}
    return {
        "callbacks": [CallbackHandler()],
        "metadata": {"langfuse_trace_name": trace_name, "langfuse_tags": tags if tags is not None else []}
    }

class HTTPLoggedException(HTTPException):
    def __init__(self,status_code:int, detail:str):
        logger.error(f"-- APP EXCEPTION >> {status_code} >> {detail}")
        super().__init__(status_code=status_code, detail=detail)