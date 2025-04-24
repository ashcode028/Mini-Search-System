import os
from typing import List

from handlers.search_engine import InMemorySearch
from models.responses import SearchResult

# Initialize search engine
search_engine = InMemorySearch()


def process_images_from_folder(folder_path: str, caption_file: str = None) -> int:
    """Process images from a folder and return embeddings and metadata"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    # Load captions if caption file is provided
    captions = {}
    if caption_file and os.path.exists(caption_file):
        with open(caption_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    image_name, caption = parts[0], parts[1]
                    captions[image_name] = caption

    # Get all image files from the folder
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    image_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        raise ValueError("No image files found in the folder")
    # Create items tuple with (caption, image_path) pairs
    items = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        caption = captions.get(image_file, "")  # Use empty string if no caption
        items.append((caption, image_path))

    # Process the batch
    search_engine.add_batch(items)
    return len(items)


def search_by_text(query: str, k: int = 5) -> List[SearchResult]:
    """Search using text query"""
    return search_engine.search_by_text(query, k)


def search_by_image(image_path: str, k: int = 5) -> List[SearchResult]:
    """Search using image query"""
    return search_engine.search_by_image(image_path, k)
