
from api.models.dto_schemas import TavilySearchConfig
from langchain_tavily import TavilySearch

def requestToTool(req: TavilySearchConfig) -> TavilySearch:
    tool = TavilySearch(
        max_results=req.max_results,
        topic=req.topic,
        include_answer=req.include_answer,
        include_raw_content=req.include_raw_content,
        include_images=req.include_images,
        include_image_descriptions=req.include_image_descriptions,
        search_depth=req.search_depth,
        time_range=req.time_range,
        start_date=req.start_date,
        end_date=req.end_date,
        include_domains=req.include_domains,
        exclude_domains=req.exclude_domains,
        include_usage= req.include_usage
    )
    return tool