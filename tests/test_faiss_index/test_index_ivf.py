import faiss
import numpy as np
import pandas as pd
from test_utils import load_data, measure_search_time

# Set up parameters
d = 512  # Dimensionality of the vectors
nq = 100  # Number of queries

if __name__ == "__main__":
    # Load sample data
    data = load_data(num_vectors=19500, dim=d)  # 19,500 vectors
    queries = load_data(num_vectors=nq, dim=d)  # 100 queries

    # Test different nlist values
    nlist_values = [10, 50, 100, 200, 500]
    results = []

    for nlist in nlist_values:
        print(f"Testing nlist={nlist}...")

        # Initialize the quantizer and the IVF index
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # Train and add data
        index.train(data)
        index.add(data)

        # Measure search time and accuracy
        search_time, distances, indices = measure_search_time(index, queries, k=5)

        # Store results
        results.append(
            {
                "nlist": nlist,
                "search_time": search_time,
                "mean_distance": np.mean(distances),
                "mean_indices": np.mean(indices),
            }
        )
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/ivf_results.csv", index=False)
