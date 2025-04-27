# FAISS Indexing Tests

This folder contains scripts to test different FAISS index types and parameters, including `IndexIVFFlat` and `IndexPQ`. The goal is to evaluate the performance and accuracy trade-offs for different configurations when datasets are huge.

## Folder Structure

- **test_index_ivf.py**: Test script for `IndexIVFFlat` with different `nlist` values.
- **test_index_pq.py**: Test script for `IndexIVFPQ` with different `nbits` values.
- **test_index_hnsw.py**: Test script for `IndexHNSWFlat` with different `M` values.
- **test_utils.py**: Helper functions for setting up the test environment, loading data, and measuring performance.
- **results/**: Stores the results of the tests.

## Requirements

You need to have the following Python packages installed:

- `faiss-cpu` (or `faiss-gpu` if using GPU version)
- `numpy`
- `pandas`
- `time`

## Running the Tests

1. Install the dependencies:
    ```bash
    pip install faiss-cpu numpy pandas
    ```

2. Run the tests for `IndexIVFFlat`:
    ```bash
    python test_index_ivf.py
    ```

3. Run the tests for `IndexIVFPQ`:
    ```bash
    python test_index_pq.py
    ```

4. Check the results in the `results/` directory.

## Results

The results will be stored in CSV files in the `results/` directory (`ivf_results.csv` and `pq_results.csv`). These files will contain search times and accuracy metrics for different configurations.

## Edge Cases to Consider

- Small datasets (e.g., `nb = 100` vectors)
- Large datasets (e.g., `nb = 1,000,000` vectors)
- Queries with all identical vectors
- Queries with vectors far from any data points
