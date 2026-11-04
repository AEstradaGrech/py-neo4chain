from pathlib import Path

from fastapi import HTTPException
from loguru import logger

from api.infrastructure.llm.statics import DATA_DIR

class HTTPLoggedException(HTTPException):
    def __init__(self,status_code:int, detail:str):
        logger.error(f"-- APP EXCEPTION >> {status_code} >> {detail}")
        super().__init__(status_code=status_code, detail=detail)

def validated_file_path(file_path: str) -> Path:
    """Absolute path of a file inside the import dir, or a tool_error.
    file_path comes from the LLM: don't let it read outside the import dir (e.g. '../.env')"""
    import_dir = Path(DATA_DIR).resolve()
    full_path_to_file = (import_dir / file_path).resolve()
    if not full_path_to_file.is_relative_to(import_dir):
        raise HTTPLoggedException(status_code=400, detail=f"File is outside the import directory: {file_path}")
    if not full_path_to_file.is_file():
        raise HTTPLoggedException(status_code=400, detail=f"File does not exist in import directory: {file_path}")
    return full_path_to_file