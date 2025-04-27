# Visual Search System

A minimal semantic search system that indexes both images(visual) and captions(text), allowing users to perform semantic searches using text and image queries. This system uses sentence-transformers for embedding generation and FAISS for efficient vector storage and retrieval.


## Features

- Image and text embedding generation using sentence-transformers
- Vector storage and search using FAISS
- REST API for text-based and image-based semantic search on images and captions
- Support for top-k matches with similarity scores

## Run app using docker Setup
Clone the repository (if not already done):
```
git clone <repo-url>
cd Semantic-Search
```
2. Build the Docker images:
In the root of the project, run the following command to build application.
```
docker-compose build

```
3. Run the containers:
Once the build is successful, start the containers using Docker Compose:
```
docker-compose up
```
This will start the server on port 8080.

4. Access the application:

The search application will be running on http://0.0.0.0:8080.

5. Clean up (optional):
To stop and remove the containers:
```
docker-compose down
```

## Run the app locally using the code

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
# Faiss package: Recommended to use conda to install faiss-cpu package
conda install -c conda-forge faiss-cpu spacy nltk
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```


3. Run the application:
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
│   ├── embedding_generator 
│   │   ├── __init__.py
│   │   ├── base_generator.py
│   │   ├── clip_embeddings.py
│   │   └── minilm_embeddings.py
│   ├── search_engine.py
│   ├── search_handler.py
│   └── search_instance.py
├── models/            # api request and response models
├── requirements.txt     # Project dependencies
├── routes/            # api routes
     └── search_apis.py
├── tests/            # Tests and benchmarking indexes
    └── test_faiss_index
        ├── README.md
        ├── results
        │   ├── hnsw_results.csv
        │   ├── ivf_results.csv
        │   └── pq_results.csv
        ├── test_index_hnsw.py
        ├── test_index_ivf.py
        ├── test_index_pq.py
        └── test_utils.py
```
## High Level Architecture
![image](https://github.com/user-attachments/assets/502d4fee-f025-4d54-a2ab-055ed4ac72d2)
