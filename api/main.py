
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_neo4j import AsyncNeo4jSaver
from langfuse import get_client
from neo4j import AsyncGraphDatabase
from api.routes.devtests_api import router as dev_router 
from api.routes.tavily_api import router as tavily_router 
from api.routes.neo4j_api import router as neo4j_router
from api.routes.kgraph_api import router as kgraph_router

from dotenv import load_dotenv
import os

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncGraphDatabase.driver(
        os.environ["NEO_URI"],
        auth=(os.environ["NEO_USERNAME"], os.environ["NEO_PASS"]),
        notifications_min_severity="OFF",  # hides the 'label/property does not exist' warnings
    ) as driver:
        saver = AsyncNeo4jSaver(driver)
        await saver.setup()   # constraints and indexes, idempotent
        app.state.neo4j_driver = driver
        app.state.neo4j_saver = saver
        yield
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

app.include_router(dev_router)
app.include_router(tavily_router)
app.include_router(neo4j_router)
app.include_router(kgraph_router)