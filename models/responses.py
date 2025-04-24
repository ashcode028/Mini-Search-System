from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel


# Base Response Models
class ErrorResponse(BaseModel):
    error: str
    message: str


class BaseResponse(BaseModel):
    status: str
    message: str


# Search Response Models
class SearchResult(BaseModel):
    image_path: str
    caption: Optional[str]
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


# Add Item Response Models
class AddItemResponse(BaseResponse):
    processed_count: int
    total_images: int


# Request Models
class AddItemRequest(BaseModel):
    folder_path: str
    caption_file: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


# Exception Handlers
def handle_not_found_error(error: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


def handle_validation_error(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def handle_internal_error(error: Exception) -> HTTPException:
    return HTTPException(status_code=500, detail=f"Internal server error: {str(error)}")


# API Response Documentation
API_RESPONSES = {
    "add_item": {
        404: {"model": ErrorResponse, "description": "Folder not found"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    "search_text": {
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    "search_image": {
        400: {"model": ErrorResponse, "description": "Invalid image"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
}
