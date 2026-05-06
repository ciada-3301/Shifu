from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
search_tool.name = "web_search"
search_tool.description = (
    "Search the web for information, documentation, code templates, or current events. "
    "Input: a search query string. Returns top web results."
)