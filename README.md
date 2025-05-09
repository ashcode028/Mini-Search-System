# Visual Search System

A minimal semantic search system that indexes both images(visual) and captions(text), allowing users to perform semantic searches using text and image queries. This system uses sentence-transformers for embedding generation and FAISS for efficient vector storage and retrieval.


## Key Features
- Seamless Search: Both text and image searches are performed efficiently, with results ranked by similarity scores for accurate and relevant results.
- Easy-to-Extend: The plug-and-play architecture allows for easy integration of new models or updates to existing models without disrupting the system.
- Multiple Indexing Strategies: Three separate FAISS indices (Text, CLIP Text, and CLIP Image) ensure efficient retrieval for both text and image queries using Euclidean and cosine similarity.

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
## High-Level Workflow

**Ingestion**: The system ingests images along with their corresponding caption paths. For simplicity, sample data has already been downloaded and stored in the `data/` folder. If you wish to use your own dataset, you can easily download and place any folder into this repository using the provided scripts. The folders are processed, and the images and captions are preprocessed to generate embeddings, which are then stored in the necessary FAISS indices and metadata DataFrame.

**Indexing**: 
- Text Index: The embeddings for the captions are stored in a Text Index, which uses Euclidean similarity to compare the embeddings. Prior to embedding generation, text cleaning and lemmatization are applied to the captions, ensuring they are properly preprocessed for better semantic matching and improved retrieval accuracy.

- CLIP Text Index: The captions are normalized before being stored in the CLIP Text Index. These captions are then converted into CLIP embeddings to enable retrieval using image queries. The indexing for this step is performed using cosine similarity, optimizing the system’s ability to match text to images.

- CLIP Image Index: The images are indexed separately in the FAISS Image Index, using cosine similarity to ensure that the most relevant images are retrieved based on image queries efficiently.

**Search**:  Users can perform searches using either a text query or an image query. The search system retrieves the top-k most relevant results by querying the appropriate indices:
- For text-based queries, it checks the Text Index and CLIP Image Index.

- For image-based queries, it searches the CLIP Image Index and CLIP Text Index.
The results are then combined and sorted by similarity scores, ensuring that the most accurate and relevant items are returned, whether the query is text or image-based.

**Persistence**: The system saves and loads its state (DataFrame and FAISS indices) to/from disk to ensure persistence across application restarts.

**Plug-and-Play Architecture**: The system is designed to be flexible, allowing you to easily plug and play different embedding generators. By simply **swapping out the embedding generator** (e.g., using a different model for text or images), the entire system will continue to function without additional modifications. This makes the system highly extensible and adaptable to future model updates or replacements.


## High Level Architecture
![image](https://github.com/user-attachments/assets/502d4fee-f025-4d54-a2ab-055ed4ac72d2)

## Improvements and Considerations
1. **Reranking Overkill**
Reranking might be seen as an overkill for this system as the use of multiple indices (for text and image queries) already ensures accurate results. Since the system retrieves relevant items based on cosine and Euclidean similarity, applying an additional reranking mechanism could unnecessarily increase complexity without significantly improving results. In most scenarios, the initial retrieval from the indices, combined with the similarity score ranking, will be sufficient to present highly relevant results, making reranking an unnecessary step.

2. **Switching to OpenAI CLIP Models**
The current implementation uses a combination of MiniLM and CLIP for generating embeddings. For even better performance, you could consider swapping out the existing models for the latest OpenAI CLIP models. These models are known for their state-of-the-art performance in both text-to-image and image-to-text retrieval tasks. Leveraging OpenAI's pre-trained CLIP models could enhance the semantic matching capability, especially in complex or large-scale datasets, making the search more precise and efficient.

3. **Alternative Indexing Methods**
Currently, FAISS uses cosine similarity to index and search for the most relevant results. However, there are several other FAISS indexing methods that can improve performance and scalability:
    - IVF (Inverted File): The IVF (Index Flat) method is a useful option when dealing with large datasets, as it groups data points into clusters and performs searches within those clusters, resulting in faster retrieval.
    - IVFPQ (Inverted File with Product Quantization): The IVFPQ method combines IVF with product quantization to reduce the size of the vectors and speed up both training and search time. This is particularly useful for large-scale datasets where memory and retrieval time are crucial.
    - HNSW (Hierarchical Navigable Small World): The HNSW method is a graph-based approach that excels in high-dimensional spaces. It ensures faster and more accurate search results, especially in situations where the dataset is highly dynamic and continuously growing.



## Contributing and Extending the System
This system is designed to be extensible, and we welcome **contributions**! If you have ideas for improving the system or would like to extend it in any way, feel free to open a pull request (PR). Whether it’s adding new features, optimizing existing code, or integrating new models, your contributions are highly encouraged.

### How to Contribute:
1. Fork the repository to your own GitHub account.

2. Clone your fork to your local machine.

3. Make changes or improvements to the system. You can:
    - Implement new embedding generators.
    
    - Add new indexing methods (e.g., IVF, IVFPQ, HNSW).
    
    - Improve existing features like the search system or persistence layer.

4. Test your changes thoroughly to ensure they don’t break existing functionality. I have also added tests for benchmarking the performance of different indexing methods in the test/ folder. You can reuse these tests to evaluate the effectiveness and performance of different indexing strategies in your own setup.

5. Submit a pull request to the main repository. Please provide a clear description of your changes and why they improve the system.
