import time

import numpy as np


def load_data(num_vectors, dim, seed=42):
    """Load or generate random data for testing."""
    np.random.seed(seed)
    data = np.random.random((num_vectors, dim)).astype("float32")
    return data


def measure_search_time(index, queries, k=5):
    """Measure the time taken for search."""
    start_time = time.time()
    distances, indices = index.search(queries, k)
    end_time = time.time()
    search_time = end_time - start_time
    return search_time, distances, indices
