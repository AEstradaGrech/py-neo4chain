
from pydantic import BaseModel, Field
from datetime import date
from typing import List, Literal, Optional


class TavilySearchConfig(BaseModel):
    """Configuration for Tavily web search."""

    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return"
    )
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="Category of the search: general, news or finance"
    )
    include_answer: bool = Field(
        default=False,
        description="Whether to include an AI-generated answer"
    )
    include_raw_content: bool = Field(
        default=False,
        description="Whether to include the raw content of search results"
    )
    include_images: bool = Field(
        default=False,
        description="Whether to include images in the search results"
    )
    include_image_descriptions: bool = Field(
        default=False,
        description="Whether to include descriptions of the returned images"
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Search depth: basic or advanced"
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description="Time range for filtering search results"
    )
    start_date: date | None = Field(
        default=None,
        description="Start date for filtering results (inclusive)"
    )
    end_date: date | None = Field(
        default=None,
        description="End date for filtering results (inclusive)"
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Domains to include in the search results"
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Domains to exclude from the search results"
    )
    include_usage: bool = Field(
        default=False,
        description="Whether to include search usage information"
    )

class TavilyExtractConfig(BaseModel):
    """Tavily extract parameters"""
    urls: List[str]=Field(description="List of urls to extract data from. Required")
    extract_depth:str = Field(description="Depth of the extraction. 'basic' | 'advanced'", default="basic"),
    include_images: bool = Field(
        default=False,
        description="Whether to include images in the extraction results"
    )
class TavilyAgentRequest(BaseModel):
    input: str = Field(description="User query")
    sysmsg: Optional[str] = Field(description="System message for the agent", default=None)
    model: Optional[str]=Field(description="Optional OLLAMA model to use with the agent. Default to app.default_llm_model", default=None)
    search_config:Optional[TavilySearchConfig]= Field(description="Tavily API params for the search endpoint", default=None)
    extract_config:Optional[TavilyExtractConfig]= Field(description="Tavily API params for the extract endpoint", default=None)

class Neo4jSchema(BaseModel):
    nodes:dict[str, str] = Field(description="KVP >> NAME : {METADATA}")
    relationships:dict[str,str] = Field(description="Node relationships")
    graph_relation:List[str] = Field(description="Current Node to Node relationships")