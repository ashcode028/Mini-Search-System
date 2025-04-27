import faiss
import numpy as np
import pandas as pd
from test_utils import load_data, measure_search_time

# Set up parameters
d = 512  # Dimensionality of the vectors
nq = 100  # Number of queries

if __name__ == "__main__":
    """
    Notes:
    The M value impacts the recall and speed of the search. Larger values may improve recall but slow down the search.
    The ef_construction parameter controls the efficiency and quality of the index construction. Higher values result in better search accuracy but slower index creation.
    """
    # Load sample data
    data = load_data(num_vectors=65536, dim=d)  # 65,536 vectors
    queries = load_data(num_vectors=nq, dim=d)  # 100 queries

    # Normalize data and queries (for Cosine similarity)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Test different M values for HNSW
    M_values = [8, 16, 32]  # Number of neighbors to explore per node
    ef_construction_values = [200, 400, 600]  # ef_construction values
    results = []

    for M in M_values:
        for ef_construction in ef_construction_values:
            print(f"Testing M={M}, ef_construction={ef_construction}...")

            # Initialize the HNSW index with Inner Product (Cosine similarity)
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = ef_construction

            # Add data to the index
            index.add(data)

            # Measure search time and accuracy
            search_time, distances, indices = measure_search_time(index, queries, k=5)

            # Store results
            results.append(
                {
                    "M": M,
                    "ef_construction": ef_construction,
                    "search_time": search_time,
                    "mean_distance": np.mean(distances),
                    "mean_indices": np.mean(indices),
                }
            )

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/hnsw_results.csv", index=False)
