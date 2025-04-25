from fastapi import Request

from handlers.search_engine import InMemorySearch

search_engine = InMemorySearch()


def get_search_engine(request: Request) -> InMemorySearch:
    return request.app.state.search_engine
