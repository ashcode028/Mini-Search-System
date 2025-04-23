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
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `data` directory and add your images and captions:
```bash
mkdir data
```

4. Run the application:
```bash
uvicorn app:app --reload
```

## API Endpoints

- `POST /search-text`: Search using text query
  - Request body: `{"query": "your search text", "k": 5}`
  - Returns: Top-k matching images with similarity scores

- `POST /search-image`: Search using image query (Bonus)
  - Request: Multipart form with image file
  - Returns: Top-k matching images with similarity scores

## Project Structure

```
.
├── app.py              # Main application file
├── data/              # Directory for images and captions
├── requirements.txt   # Project dependencies
└── README.md         # This file
```
