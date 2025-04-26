# Semantic Search System

A minimal semantic search system that indexes images and text descriptions, allowing users to perform semantic searches using text queries.

## Features

- Image and text embedding generation using sentence-transformers
- Vector storage using FAISS
- REST API for text-based and image-based semantic search
- Support for top-k matches with similarity scores

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# optional conda env
# conda activate <your_env>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# This causes issues sometimes, it is recommended to use conda to install faiss-cpu package
pip uninstall faiss-cpu
conda install -c conda-forge faiss-cpu
```

3. Create a `data` directory and add your images and captions:
```bash
mkdir data
```

4. Run the application:
```bash
python app.py
```

## API Endpoints
- `POST /ingest-data`: Api to upload images and captions
  - Request body: `{ "folder_path": "your/images", "caption_file": "your/captions/sample_captions.txt"}`
  - Returns: No of processed files

- `POST /search-text`: Search using text query
  - Request body: `{"query": "your search text", "k": 5}`
  - Returns: Top-k matching images with similarity scores

- `POST /search-image`: Search using image query (Bonus)
  - Request: Multipart form with image file
  - Returns: Top-k matching images with similarity scores

## Project Structure

```
├── README.md
├── app.py                # Main application file
├── data/                 #  Directory for images and captions
├── handlers/            # api handlers
│   ├── __init__.py
│   ├── search_engine.py
│   └── search_handler.py
├── models/            # api request and response models
├── requirements.txt     # Project dependencies
├── routes/            # api routes
     └── search_apis.py
```
