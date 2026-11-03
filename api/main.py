
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_neo4j import AsyncNeo4jSaver
from langfuse import get_client
from neo4j import AsyncGraphDatabase, GraphDatabase
from api.infrastructure.repositories.neo4j.neo4j_utils import set_kg_driver, set_kg_sync_driver
from api.routes.tavily_api import router as tavily_router 
from api.routes.neo4j_api import router as neo4j_router
from api.routes.kgraph_api import router as kgraph_router

from dotenv import load_dotenv
import os

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with (
            AsyncGraphDatabase.driver(
                os.environ["NEO_URI"],
                auth=(os.environ["NEO_USERNAME"], os.environ["NEO_PASS"]),
                notifications_min_severity="OFF",  # hides the 'label/property does not exist' warnings
            ) as driver,
            AsyncGraphDatabase.driver(
                os.environ["NEODB_URI"],
                auth=(os.environ["NEO_USERNAME"], os.environ["NEO_PASS"]),
            ) as kg_driver,
        ):
            # sync driver required by SimpleKGPipeline (its Neo4jWriter doesn't await)
            with GraphDatabase.driver(
                os.environ["NEODB_URI"],
                auth=(os.environ["NEO_USERNAME"], os.environ["NEO_PASS"]),
            ) as kg_sync_driver:
                saver = AsyncNeo4jSaver(driver)
                await saver.setup()   # constraints and indexes, idempotent
                await kg_driver.verify_connectivity()
                kg_sync_driver.verify_connectivity()
                app.state.neo4j_driver = driver
                app.state.neo4j_saver = saver
                app.state.neodb_driver = kg_driver
                app.state.neodb_sync_driver = kg_sync_driver
                set_kg_driver(kg_driver)
                set_kg_sync_driver(kg_sync_driver)
                yield
                # shutdown: clear the globals before the drivers get closed
                set_kg_driver(None)
                set_kg_sync_driver(None)
    finally:
        # Langfuse exports traces in background batches: flush the pending ones before the process exits
        get_client().shutdown()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(tavily_router)
app.include_router(neo4j_router)
app.include_router(kgraph_router)