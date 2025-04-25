import os
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

    def search_images_caption_by_text(
        self, query: str, k: int = 5
    ) -> List[SearchResult]:
        """Search using text query and return results from both text and image indices"""
        # Generate embedding for the query text
        query_embedding = self.generate_text_embedding(query)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        # Search using the text index
        text_results = self._build_results(indices=indices, distances=distances)

        # Generate image embedding for the query text
        image_embedding = self.image_model.encode(query)

        # Search using the image index
        distances, indices = self.image_index.search(np.array([image_embedding]), k)
        image_results = self._build_results(indices=indices, distances=distances)

        # Combine results from both text and image search
        all_results = text_results + image_results

        # Sort the results by score (higher score means better match)
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Return the top k results (from both text and image matches)
        return all_results[:k]

    def search_by_text_query(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search using text query"""
        query_embedding = self.generate_text_embedding(query)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        return self._build_results(indices=indices, distances=distances)

    def search_by_image_query(self, image_path: str, k: int = 5) -> List[SearchResult]:
        """Search using image query"""
        query_embedding = self.generate_image_embedding(image_path)
        distances, indices = self.image_index.search(np.array([query_embedding]), k)
        return self._build_results(indices=indices, distances=distances)

    def _build_results(self, indices, distances) -> List[SearchResult]:
        """Construct SearchResult objects from indices and distances"""
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

    def save(self, data_dir: str) -> None:
        """Save dataframe (without embeddings) and embeddings to disk"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Drop embeddings before saving df
        df_to_save = self.df.drop(columns=["text_embedding", "image_embedding"])
        df_to_save.to_parquet(os.path.join(data_dir, "dataframe.parquet"))

        # Save embeddings
        if self.text_index.ntotal > 0:
            text_embeddings = self.text_index.reconstruct_n(0, self.text_index.ntotal)
            np.save(os.path.join(data_dir, "text_embeddings.npy"), text_embeddings)

        if self.image_index.ntotal > 0:
            image_embeddings = self.image_index.reconstruct_n(
                0, self.image_index.ntotal
            )
            np.save(os.path.join(data_dir, "image_embeddings.npy"), image_embeddings)

    def load(self, data_dir: str) -> None:
        """Load dataframe and embeddings from disk and reconstruct internal state"""
        if not os.path.exists(data_dir):
            return

        # Load dataframe (without embeddings)
        df_path = os.path.join(data_dir, "dataframe.parquet")
        if os.path.exists(df_path):
            self.df = pd.read_parquet(df_path)
        else:
            self.df = pd.DataFrame()

        # Load embeddings
        text_embeddings_file = os.path.join(data_dir, "text_embeddings.npy")
        image_embeddings_file = os.path.join(data_dir, "image_embeddings.npy")

        text_embeddings = None
        image_embeddings = None

        if os.path.exists(text_embeddings_file):
            text_embeddings = np.load(text_embeddings_file)
            self.text_index.add(text_embeddings)

        if os.path.exists(image_embeddings_file):
            image_embeddings = np.load(image_embeddings_file)
            self.image_index.add(image_embeddings)

        # Attach embeddings back into df
        if text_embeddings is not None and image_embeddings is not None:
            # Sanity check
            assert len(self.df) == len(text_embeddings) == len(image_embeddings)
            self.df["text_embedding"] = list(text_embeddings)
            self.df["image_embedding"] = list(image_embeddings)
