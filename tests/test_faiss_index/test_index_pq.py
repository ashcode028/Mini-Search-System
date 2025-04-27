import faiss
import numpy as np
import pandas as pd
from test_utils import load_data, measure_search_time

# Set up parameters
d = 512  # Dimensionality of the vectors
nq = 100  # Number of queries

if __name__ == "__main__":
    """
    nbits controls how many bits are used to store the quantized representation for each subvector.
    nbits_values tests how different levels of quantization affect the search performance in terms of speed and accuracy.
    Lower nbits values will result in faster searches but might sacrifice accuracy.
    Higher nbits values will increase accuracy (recall) but may slow down the search and increase memory usage.
    """
    # Load sample data
    data = load_data(num_vectors=65536, dim=d)  # 65,536 vectors
    queries = load_data(num_vectors=nq, dim=d)  # 100 queries

    # Test different nbits values
    nbits_values = [4, 8, 16]
    results = []

    for nbits in nbits_values:
        print(f"Testing nbits={nbits}...")

        # Initialize the PQ index
        nsubquantizers = 16  # Number of subquantizers
        index = faiss.IndexIVFPQ(d, nsubquantizers, nbits)

        # Train and add data
        index.train(data)
        index.add(data)

        # Measure search time and accuracy
        search_time, distances, indices = measure_search_time(index, queries, k=5)

        # Store results
        results.append(
            {
                "nbits": nbits,
                "search_time": search_time,
                "mean_distance": np.mean(distances),
                "mean_indices": np.mean(indices),
            }
        )

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/pq_results.csv", index=False)
