from fastapi import APIRouter, UploadFile, File
from handlers.search_handler import (
    process_images_from_folder,
    add_images_to_index,
    search_by_text,
    search_by_image
)
from models.responses import (
    AddItemRequest,
    AddItemResponse,
    SearchRequest,
    SearchResponse,
    API_RESPONSES,
    handle_not_found_error,
    handle_validation_error,
    handle_internal_error
)
import os
from PIL import Image

router = APIRouter()

@router.post(
    "/add-item",
    response_model=AddItemResponse,
    responses=API_RESPONSES["add_item"]
)
async def add_item(request: AddItemRequest):
    """Add multiple images from a folder to the index"""
    try:
        embeddings_list, metadata_list, processed_count = process_images_from_folder(
            request.folder_path, request.caption_file
        )
        add_images_to_index(embeddings_list, metadata_list)
        
        return AddItemResponse(
            status="success",
            message=f"Successfully processed {processed_count} images",
            processed_count=processed_count,
            total_images=len(os.listdir(request.folder_path))
        )
    except FileNotFoundError as e:
        raise handle_not_found_error(e)
    except ValueError as e:
        raise handle_validation_error(e)
    except Exception as e:
        raise handle_internal_error(e)

@router.post(
    "/search-text",
    response_model=SearchResponse,
    responses=API_RESPONSES["search_text"]
)
async def search_text(request: SearchRequest):
    """Search using text query"""
    try:
        results = search_by_text(request.query, request.k)
        return SearchResponse(results=results)
    except Exception as e:
        raise handle_internal_error(e)
