import os

from fastapi import APIRouter, Depends, File, UploadFile

from handlers.search_handler import index_data, process_images_from_folder
from handlers.search_instance import get_search_engine
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
    "/ingest-data",
    response_model=AddItemResponse,
    responses=API_RESPONSES["ingest-data"],
)
async def ingest_data(
    request: AddItemRequest, search_engine=Depends(get_search_engine)
):
    """
    Add multiple images,captions from a folder to the search indexes.

    This endpoint processes images from a given folder, adds them to the search index,
    and returns the number of successfully processed images.

    :param request: Request body containing folder path and caption file.
    :param search_engine: Dependency that provides access to the search engine instance.
    :return: A response with the status and count of processed images.
    :raises: FileNotFoundError if the folder does not exist,
             ValueError if no images are found in the folder,
             Internal server errors for any unexpected issues.
    """
    try:
        items, processed_count = process_images_from_folder(
            folder_path=request.folder_path,
            caption_file=request.caption_file,
        )
        index_data(search_engine=search_engine, items=items)

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
async def search_text(request: SearchRequest, search_engine=Depends(get_search_engine)):
    """
     Perform image/captions search using an image query.

    This endpoint allows users to search the index using a text query and the system will return a list
    of similar images and their captions with similar scores.

    :param request: Request body containing the search query and the number of results (k).
    :param search_engine: Dependency that provides access to the search engine instance.
    :return: A response with a list of search results.
    :raises: Internal server error for any unexpected issues.
    """
    try:
        results = search_engine.search_images_caption_by_text(request.query, request.k)
        return SearchResponse(results=results)
    except Exception as e:
        raise handle_internal_error(e)


@router.post(
    "/search-image",
    response_model=SearchResponse,
    responses=API_RESPONSES["search_image"],
)
async def search_image(
    image: UploadFile = File(...), k: int = 5, search_engine=Depends(get_search_engine)
):
    """
    Perform image/captions search using an image query.

    This endpoint allows users to upload an image, and the system will return a list
    of similar images and their captions with similar scores.

    :param image: The uploaded image file to search with.
    :param k: The number of similar images to return (default is 5).
    :param search_engine: Dependency that provides access to the search engine instance.
    :return: A response with a list of search results based on the image query.
    :raises: Internal server error for any unexpected issues,
             and ensures cleanup of temporary files in case of failure.
    """
    # Save to temporary path
    temp_path = "temp_image.jpg"
    try:
        with open(temp_path, "wb") as f:
            f.write(await image.read())
        # Search using the image
        results = search_engine.search_images_caption_by_image(temp_path, k)

        # Clean up
        os.remove(temp_path)
        return SearchResponse(results=results)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise handle_internal_error(e)
