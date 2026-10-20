from fastapi import HTTPException
from langchain_ollama import ChatOllama
from loguru import logger

from api.utils.statics import default_llm_model

def ollama(model:str = None, temp:float = 0.7, max_tokens:int = 600, ctx_len:int = 4096, verbose:bool = False) -> ChatOllama:
    logger.warning(f"-- OLLAMA INSTANCE FOR: {default_llm_model if model is None else model}")
    return ChatOllama(model=default_llm_model if model is None else model, temperature=temp, num_predict=max_tokens, num_ctx=ctx_len, verbose=verbose)
    
class HTTPLoggedException(HTTPException):
    def __init__(self,status_code:int, detail:str):
        logger.error(f"-- APP EXCEPTION >> {status_code} >> {detail}")
        super().__init__(status_code=status_code, detail=detail)