import numpy as np
from sentence_transformers import SentenceTransformer

from handlers.embedding_generator.base_generator import EmbeddingGenerator


class MiniLMGenerator(EmbeddingGenerator):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384

    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def encode_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError("This model does not support image encoding")
