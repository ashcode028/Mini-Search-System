import os

from fastapi import APIRouter, File, UploadFile

from handlers.search_handler import (
    process_images_from_folder,
    search_by_image,
    search_by_text,
)
from models.responses import (
    API_RESPONSES,
    AddItemRequest,
    AddItemResponse,
    SearchRequest,
    SearchResponse,
    handle_internal_error,
    handle_not_found_error,
    handle_validation_error,
)

router = APIRouter()


@router.post(
    "/add-item", response_model=AddItemResponse, responses=API_RESPONSES["add_item"]
)
async def add_item(request: AddItemRequest):
    """Add multiple images from a folder to the index"""
    try:
        processed_count = process_images_from_folder(
            request.folder_path, request.caption_file
        )
        return AddItemResponse(
            status="success",
            message=f"Successfully processed {processed_count} images",
            processed_count=processed_count,
            total_images=len(os.listdir(request.folder_path)),
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
    responses=API_RESPONSES["search_text"],
)
async def search_text(request: SearchRequest):
    """Search using text query"""
    try:
        results = search_by_text(request.query, request.k)
        return SearchResponse(results=results)
    except Exception as e:
        raise handle_internal_error(e)


@router.post(
    "/search-image",
    response_model=SearchResponse,
    responses=API_RESPONSES["search_image"],
)
async def search_image(image: UploadFile = File(...), k: int = 5):
    """Search using image query"""
    try:
        # Save to temporary path
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(await image.read())
        # Search using the image
        results = search_by_image(temp_path, k)

        # Clean up
        os.remove(temp_path)

        return SearchResponse(results=results)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise handle_internal_error(e)
