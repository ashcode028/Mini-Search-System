from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

from models.responses import SearchResult


class InMemorySearch:
    def __init__(self):
        """Initialize the in-memory search system with separate indices for text and images"""
        # Initialize models
        self.df = pd.DataFrame()
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.image_model = SentenceTransformer("clip-ViT-B-32")
        # Get embedding dimensions
        self.text_dimension = 384
        self.image_dimension = 512

        # Initialize FAISS indices
        self.text_index = faiss.IndexFlatL2(self.text_dimension)
        self.image_index = faiss.IndexFlatL2(self.image_dimension)

        # Store metadata for both indices
        self.metadata = []

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.text_model.encode(text)

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for image"""
        image = Image.open(image_path)
        return self.image_model.encode(image)

    def add_item(self, text: str, image_path: str) -> int:
        """Add a new item with both text and image"""
        try:
            # Generate embeddings
            text_embedding = self.generate_text_embedding(text)
            image_embedding = self.generate_image_embedding(image_path)

            # Create new ID
            new_id = len(self.df)

            # Add to DataFrame
            self.df.loc[new_id] = {
                "text": text,
                "image_path": image_path,
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
            }

            # Add to FAISS indices
            self.text_index.add(np.array([text_embedding]))
            self.image_index.add(np.array([image_embedding]))

            return new_id
        except Exception as _:
            raise

    def add_batch(self, items: List[Tuple[str, str]]) -> List[int]:
        """Add multiple items in batch"""
        try:
            ids = []
            texts = []
            image_paths = []
            text_embeddings = []
            image_embeddings = []

            for text, image_path in items:
                # Generate embeddings
                text_embedding = self.generate_text_embedding(text)
                image_embedding = self.generate_image_embedding(image_path)

                # Store data
                new_id = len(self.df) + len(ids)
                ids.append(new_id)
                texts.append(text)
                image_paths.append(image_path)
                text_embeddings.append(text_embedding)
                image_embeddings.append(image_embedding)

            # Add to DataFrame
            batch_df = pd.DataFrame(
                {
                    "text": texts,
                    "image_path": image_paths,
                    "text_embedding": text_embeddings,
                    "image_embedding": image_embeddings,
                },
                index=ids,
            )
            self.df = pd.concat([self.df, batch_df])

            # Add to FAISS indices
            self.text_index.add(np.array(text_embeddings))
            self.image_index.add(np.array(image_embeddings))

            return ids
        except Exception as _:
            raise

    def search_by_text(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search using text query"""
        query_embedding = self.generate_text_embedding(query)
        return self._search_text(query_embedding, k)

    def search_by_image(self, image_path: str, k: int = 5) -> List[SearchResult]:
        """Search using image query"""
        query_embedding = self.generate_image_embedding(image_path)
        return self._search_image(query_embedding, k)

    def _search_text(self, query_embedding: np.ndarray, k: int) -> List[SearchResult]:
        """Internal method for text search"""
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.df):
                caption, image_path = self.df.iloc[idx][:2]
                results.append(
                    SearchResult(
                        image_path=image_path,
                        caption=caption,
                        score=float(1 / (1 + distance)),
                    )
                )
        return results

    def _search_image(self, query_embedding: np.ndarray, k: int) -> List[SearchResult]:
        """Internal method for image search"""
        distances, indices = self.image_index.search(np.array([query_embedding]), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.df):
                caption, image_path = self.df.iloc[idx][:2]
                results.append(
                    SearchResult(
                        image_path=image_path,
                        caption=caption,
                        score=float(1 / (1 + distance)),
                    )
                )
        return results

    # def save(self, data_dir: str) -> None:
    #     """Save both indices and metadata to disk"""
    #     if not os.path.exists(data_dir):
    #         os.makedirs(data_dir)
    #
    #     # Save metadata
    #     with open(os.path.join(data_dir, "metadata.json"), "w") as f:
    #         json.dump(self.metadata, f)
    #
    #     # Save embeddings
    #     if self.text_index.ntotal > 0:
    #         text_embeddings = self.text_index.reconstruct_n(0, self.text_index.ntotal)
    #         np.save(os.path.join(data_dir, "text_embeddings.npy"), text_embeddings)
    #
    #     if self.image_index.ntotal > 0:
    #         image_embeddings = self.image_index.reconstruct_n(
    #             0, self.image_index.ntotal
    #         )
    #         np.save(os.path.join(data_dir, "image_embeddings.npy"), image_embeddings)
    #
    # def load(self, data_dir: str) -> None:
    #     """Load both indices and metadata from disk"""
    #     if not os.path.exists(data_dir):
    #         return
    #
    #     # Load metadata
    #     metadata_file = os.path.join(data_dir, "metadata.json")
    #     if os.path.exists(metadata_file):
    #         with open(metadata_file, "r") as f:
    #             self.metadata = json.load(f)
    #
    #     # Load embeddings
    #     text_embeddings_file = os.path.join(data_dir, "text_embeddings.npy")
    #     if os.path.exists(text_embeddings_file):
    #         text_embeddings = np.load(text_embeddings_file)
    #         self.text_index.add(text_embeddings)
    #
    #     image_embeddings_file = os.path.join(data_dir, "image_embeddings.npy")
    #     if os.path.exists(image_embeddings_file):
    #         image_embeddings = np.load(image_embeddings_file)
    #         self.image_index.add(image_embeddings)
