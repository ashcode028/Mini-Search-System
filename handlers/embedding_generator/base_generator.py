from abc import ABC, abstractmethod

import numpy as np


class EmbeddingGenerator(ABC):
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        """Generate embedding for image."""
        pass
