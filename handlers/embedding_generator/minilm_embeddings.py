import re
import string

import numpy as np
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from handlers.embedding_generator.base_generator import EmbeddingGenerator


class MiniLMGenerator(EmbeddingGenerator):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        self.nlp = spacy.load("en_core_web_sm")

    def _preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and punctuation
        text = re.sub(f"[{string.punctuation}]", "", text)

        # Tokenize and remove stopwords
        tokens = text.split()
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemma-tize the tokens
        lemmatized_tokens = [self.nlp(word)[0].lemma_ for word in tokens]

        # Rejoin into a single string
        clean_text = " ".join(lemmatized_tokens)

        return clean_text

    def encode_text(self, text: str) -> np.ndarray:
        clean_text = self._preprocess_text(text)
        return self.model.encode(clean_text)

    def encode_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError("This model does not support image encoding")
