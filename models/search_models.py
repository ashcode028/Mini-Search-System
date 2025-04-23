from typing import List, Optional
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResponse(BaseModel):
    results: List[dict]

class AddItemRequest(BaseModel):
    folder_path: str
    caption_file: Optional[str] = None 