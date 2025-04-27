import os
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd

from handlers.embedding_generator.clip_embeddings import CLIPEmbeddingGenerator
from handlers.embedding_generator.minilm_embeddings import MiniLMGenerator
from models.responses import SearchResult


class InMemorySearch:
    def __init__(self):
        """Initialize the in-memory search system with separate indices for text and images"""

        self.df = pd.DataFrame()
        self.text_encoder = MiniLMGenerator()
        self.clip_encoder = CLIPEmbeddingGenerator()
        # Initialize models
        self.image_model = self.clip_encoder.model
        self.text_model = self.text_encoder.model

        # Get embedding dimensions
        self.text_dimension = self.text_encoder.dimension
        self.image_dimension = self.clip_encoder.dimension

        # Initialize FAISS indices
        self.text_index = faiss.IndexFlatL2(self.text_dimension)
        self.image_index = faiss.IndexFlatL2(self.image_dimension)
        self.clip_text_index = faiss.IndexFlatL2(self.image_dimension)

    def add_item(self, text: str, image_path: str) -> int:
        """Add a new item with both text and image"""
        try:
            # Generate embeddings
            text_embedding = self.text_encoder.encode_text(text)
            image_embedding = self.clip_encoder.encode_image(image_path)
            clip_text_embedding = self.clip_encoder.encode_text(text)
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
            self.clip_text_index.add(np.array([clip_text_embedding]))
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
            clip_text_embeddings = []

            for text, image_path in items:
                # Generate embeddings
                text_embedding = self.text_encoder.encode_text(text)
                image_embedding = self.clip_encoder.encode_image(image_path)
                clip_text_embedding = self.clip_encoder.encode_text(text)

                # Store data
                new_id = len(self.df) + len(ids)
                ids.append(new_id)
                texts.append(text)
                image_paths.append(image_path)
                text_embeddings.append(text_embedding)
                image_embeddings.append(image_embedding)
                clip_text_embeddings.append(clip_text_embedding)

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
            self.clip_text_index.add(np.array(clip_text_embeddings))
            self.image_index.add(np.array(image_embeddings))

            return ids
        except Exception as _:
            raise

    def search_images_caption_by_text(
        self, query: str, k: int = 5
    ) -> List[SearchResult]:
        """Search using text query and return results from both text and image indices"""
        # Search using the text index
        text_results = self.search_captions_by_text_query(query=query, k=k)
        # Search using the image index
        image_results = self.search_images_by_text_query(query=query, k=k)

        # Combine results from both text and image search
        all_results = text_results + image_results

        # Sort the results by score (higher score means better match)
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Return the top k results (from both text and image matches)
        return all_results[:k]

    def search_images_caption_by_image(
        self, image_path: str, k: int = 5
    ) -> List[SearchResult]:
        """Search using image query and return results from both text and image indices"""
        # Search using the text index
        text_results = self.search_captions_by_image_query(image_path=image_path, k=k)
        # Search using the image index
        image_results = self.search_images_by_image_query(image_path=image_path, k=k)

        # Combine results from both text and image search
        all_results = text_results + image_results

        # Sort the results by score (higher score means better match)
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Return the top k results (from both text and image matches)
        return all_results[:k]

    def search_captions_by_text_query(
        self, query: str, k: int = 5
    ) -> List[SearchResult]:
        """Search using text query"""
        query_embedding = self.text_encoder.encode_text(query)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        return self._build_results(indices=indices, distances=distances)

    def search_images_by_text_query(self, query: str, k: int = 5) -> List[SearchResult]:
        image_embedding = self.clip_encoder.encode_text(query)
        distances, indices = self.image_index.search(np.array([image_embedding]), k)
        return self._build_results(indices=indices, distances=distances)

    def search_images_by_image_query(
        self, image_path: str, k: int = 5
    ) -> List[SearchResult]:
        """Search using image query"""
        query_embedding = self.clip_encoder.encode_image(image_path)
        distances, indices = self.image_index.search(np.array([query_embedding]), k)
        return self._build_results(indices=indices, distances=distances)

    def search_captions_by_image_query(
        self, image_path: str, k: int = 5
    ) -> List[SearchResult]:
        """Search using image query"""
        text_embedding = self.clip_encoder.encode_image(image_path)
        distances, indices = self.clip_text_index.search(np.array([text_embedding]), k)
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
        """Persist dataframe (without embeddings) and embeddings to disk"""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Drop embeddings before saving df
        df_to_save = self.df.drop(columns=["text_embedding", "image_embedding"])
        df_to_save.to_parquet(os.path.join(data_dir, "dataframe.parquet"))

        # Save FAISS indexes
        text_index_path = os.path.join(data_dir, "text.index")
        clip_text_index_path = os.path.join(data_dir, "clip_text.index")
        image_index_path = os.path.join(data_dir, "image.index")

        faiss.write_index(self.text_index, text_index_path)
        faiss.write_index(self.clip_text_index, clip_text_index_path)
        faiss.write_index(self.image_index, image_index_path)

    def load(self, data_dir: str) -> None:
        """Load dataframe and FAISS indexes from disk and reconstruct internal state"""
        if not os.path.exists(data_dir):
            return

        # Load dataframe (without embeddings)
        df_path = os.path.join(data_dir, "dataframe.parquet")
        self.df = (
            pd.read_parquet(df_path) if os.path.exists(df_path) else pd.DataFrame()
        )

        # Load FAISS indexes
        text_index_path = os.path.join(data_dir, "text.index")
        clip_text_index_path = os.path.join(data_dir, "clip_text.index")
        image_index_path = os.path.join(data_dir, "image.index")

        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)

        if os.path.exists(clip_text_index_path):
            self.clip_text_index = faiss.read_index(clip_text_index_path)

        if os.path.exists(image_index_path):
            self.image_index = faiss.read_index(image_index_path)

        assert (
            len(self.df)
            == self.text_index.ntotal
            == self.image_index.ntotal
            == self.clip_text_index.ntotal
        ), "Data mismatch between DataFrame and FAISS indexes"
