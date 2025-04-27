import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from handlers.embedding_generator.base_generator import EmbeddingGenerator


class CLIPEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self):
        self.model = SentenceTransformer("clip-ViT-B-32")
        self.dimension = 512

    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return self.model.encode(image, normalize_embeddings=True)
