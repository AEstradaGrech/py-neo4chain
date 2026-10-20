from fastapi import HTTPException
from loguru import logger

class HTTPLoggedException(HTTPException):
    def __init__(self,status_code:int, detail:str):
        logger.error(f"-- APP EXCEPTION >> {status_code} >> {detail}")
        super().__init__(status_code=status_code, detail=detail)