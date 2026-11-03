# api/infrastructure/db/neo4j_drivers.py
from neo4j import AsyncDriver

_kg_driver: AsyncDriver | None = None

def set_kg_driver(driver: AsyncDriver | None) -> None:
    global _kg_driver
    _kg_driver = driver

def kg_driver() -> AsyncDriver:
    if _kg_driver is None:
        raise RuntimeError("KG driver not initialized: it's created in the FastAPI lifespan")
    return _kg_driver