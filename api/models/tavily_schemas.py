
from pydantic import BaseModel, Field
from datetime import date
from typing import Any, Dict, List, Literal, Optional


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
    include_images: bool = Field(description="Whether to include images in the extraction results", default=False)

class TavilyMapCrawlConfig(BaseModel):
    """Tavily map and crawl parameters"""
    max_depth: Optional[int] = Field(
        description="Maximum depth of the crawl. Must be greater than 0. Default is 1",
        default=None
    )
    max_breadth: Optional[int] = Field(
        description="Maximum number of links to follow per level (per page). Must be greater than 0. Default is 20",
        default=None
    )
    limit: Optional[int] = Field(
        description="Total number of links to process before stopping. Must be greater than 0. Default is 50",
        default=None
    )
    instructions: Optional[str] = Field(
        description="Natural language instructions for the crawler. Example: 'Python SDK'",
        default=None
    )
    select_paths: Optional[List[str]] = Field(
        description="Regex patterns to select URLs with specific path patterns. Example: ['/api/v1.*']",
        default=None
    )
    select_domains: Optional[List[str]] = Field(
        description="Regex patterns to select URLs from specific domains or subdomains. Example: ['^docs\\.example\\.com$']",
        default=None
    )
    exclude_paths: Optional[List[str]] = Field(
        description="Regex patterns to exclude URLs with specific path patterns. Example: ['/private/.*', '/admin/.*']",
        default=None
    )
    exclude_domains: Optional[List[str]] = Field(
        description="Regex patterns to exclude specific domains or subdomains from mapping. Example: ['^private\\.example\\.com$']",
        default=None
    )
    allow_external: Optional[bool] = Field(
        description="Whether to follow links to external domains",
        default=None
    )

class TavilyResearchConfig(BaseModel):
    """Tavily research parameters"""
    research_model: Optional[Literal["mini", "pro", "auto"]] = Field(
        default="auto",
        alias="model",
        description="Model used by the research agent: 'mini' | 'pro' | 'auto'. Default is 'auto'"
    )
    research_output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema object defining the structure of the research output"
    )
    stream_results: Optional[bool] = Field(
        default=None,
        description="Whether to stream research results as they are generated via SSE. Default is False"
    )
    citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = Field(
        default=None,
        description="Citation format for the research report: 'numbered' | 'mla' | 'apa' | 'chicago'. Default is 'numbered'"
    )
    
class TavilyAgentRequest(BaseModel):
    input: str = Field(description="User query")
    sysmsg: Optional[str] = Field(description="System message for the agent", default=None)
    model: Optional[str]=Field(description="Optional OLLAMA model to use with the agent. Default to app.default_llm_model", default=None)
    search_config:Optional[TavilySearchConfig]= Field(description="Tavily API params for the search endpoint", default=None)
    extract_config:Optional[TavilyExtractConfig]= Field(description="Tavily API params for the extract endpoint", default=None)
    map_config:Optional[TavilyMapCrawlConfig]= Field(description="Tavily API params for the map or crawl endpoint", default=None)
    crawl_config:Optional[TavilyMapCrawlConfig]= Field(description="Tavily API params for the map or crawl endpoint", default=None)
    research_config:Optional[TavilyResearchConfig]= Field(description="Tavily API params for the research endpoint", default=None)

class Neo4jSchema(BaseModel):
    nodes:dict[str, str] = Field(description="KVP >> NAME : {METADATA}")
    relationships:dict[str,str] = Field(description="Node relationships")
    graph_relation:List[str] = Field(description="Current Node to Node relationships")
    summary:str = Field(description="Full text with the schema data")

class TextFileVectorSearch(BaseModel):
    textfile_path:str = Field(description="Path to the text file to load and embed, without file extension")
    query:str = Field(description="User input to search in the Vector store")
    chunk_size: Optional[int] = Field(description="Length of the file chunks", default=None)
    chunk_overlap: Optional[int] = Field(description="text overlap between chunks", default=None)
    k:Optional[int] = Field(description="Number of K-neighbors to return for the given input", default=None)
    min_score: Optional[float] = Field(description="Minimum score for a chunk to be included as a result", default=None)