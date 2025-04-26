import os
from typing import List, Tuple

from handlers.search_engine import InMemorySearch


def index_data(search_engine: InMemorySearch, items: List[Tuple[str, str]]) -> None:
    """
    Indexes the provided images and captions into the search engine in batches.

    :param search_engine: Instance of InMemorySearch to index data.
    :param items: List of tuples containing (caption, image_path) pairs.
    """
    batch_size = 20
    total_items = len(items)
    num_batches = (total_items // batch_size) + (
        1 if total_items % batch_size != 0 else 0
    )
    for i in range(num_batches):
        batch_items = items[i * batch_size : (i + 1) * batch_size]  # Get the batch
        search_engine.add_batch(batch_items)
        print(f"Processed batch {i + 1}/{num_batches} with {len(batch_items)} items")

    search_engine.save(f"data/sample_metadata")


def process_images_from_folder(folder_path: str, caption_file: str) -> (int, int):
    """
    Processes images from a folder and returns a list of (caption, image_path) pairs and the total count.

    :param folder_path: Path to the folder containing images.
    :param caption_file: Path to the file containing captions (tab-separated: image_name, caption).
    :return: A tuple containing a list of (caption, image_path) pairs and the total number of items.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    # Load captions
    captions = {}
    if os.path.exists(caption_file):
        with open(caption_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
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

    # Return total number of processed items
    return items, len(items)
