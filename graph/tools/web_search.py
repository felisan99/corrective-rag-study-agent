import os

from langchain_tavily import TavilySearch

# .env names this var `TAVILY` rather than the `TAVILY_API_KEY` that
# langchain-tavily looks for by default, so it's passed explicitly.
web_search = TavilySearch(max_results=3, tavily_api_key=os.environ.get("TAVILY"))
