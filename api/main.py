
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.devtests_api import router as dev_router 
from api.routes.tavily_api import router as tavily_router 
from api.routes.neo4j_api import router as neo4j_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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
