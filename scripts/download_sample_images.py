import os
from io import BytesIO

import requests
from PIL import Image

# Sample image URLs from Unsplash (free to use)
IMAGE_URLS = {
    "image1.jpg": "https://images.unsplash.com/photo-1508098682722-e99c43a406b2",
    "image2.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "image3.jpg": "https://images.unsplash.com/photo-1525845859779-54d477ff291f",
    "image4.jpg": "https://images.unsplash.com/photo-1519501025264-65ba15a82390",
    "image5.jpg": "https://images.unsplash.com/photo-1505576399279-565b52d4ac71",
    "image6.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
    "image7.jpg": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b",
    "image8.jpg": "https://images.unsplash.com/photo-1494905998402-395d579af36f",
    "image9.jpg": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e",
    "image10.jpg": "https://images.unsplash.com/photo-1512568400610-62da28bc8a13",
}


def download_images():
    """Download sample images from Unsplash"""
    # Create images directory if it doesn't exist
    os.makedirs("data/images", exist_ok=True)

    # Download each image
    for filename, url in IMAGE_URLS.items():
        try:
            # Get the image
            response = requests.get(url)
            response.raise_for_status()

            # Open and save the image
            img = Image.open(BytesIO(response.content))
            img.save(f"data/images/{filename}")
            print(f"Downloaded {filename}")

        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")


if __name__ == "__main__":
    download_images()
