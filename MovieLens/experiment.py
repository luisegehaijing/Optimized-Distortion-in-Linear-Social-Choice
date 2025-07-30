import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import main
import time
from scipy.sparse import csr_matrix
import collections
import argparse

# Ensure the distortion_model.py is in the same directory or on the Python path
# If you save the first code block as distortion_model.py, this import will work.
try:
    from main import load_and_sample_movielens_data, Instance
except ImportError:
    print("Error: Could not import 'load_and_sample_movielens_data' or 'Instance' from 'distortion_model.py'.")
    print("Please ensure the first code block is saved as 'distortion_model.py' in the same directory.")
    sys.exit(1)

def load_full_movielens_data(data_path: str):
    """
    Loads the entire MovieLens dataset into a sparse matrix.
    Expected format: user_id,item_id,rating,...

    Args:
        data_path (str): Path to the MovieLens dataset file (e.g., "u.data").

    Returns:
        tuple: A tuple containing:
            - rating_matrix (csr_matrix): Sparse matrix of all ratings.
            - user_to_idx (dict): Mapping from original user ID to internal user index.
            - item_to_idx (dict): Mapping from original item ID to internal item index.
    """
    print(f"Loading full MovieLens data from {data_path}...")
    all_ratings = []
    unique_users = set()
    unique_items = set()

    with open(data_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split(",")
                user_id, item_id, rating = int(parts[0]), int(parts[1]), float(parts[2])
                if user_id >2000:
                    break
                all_ratings.append((user_id, item_id, rating))
                unique_users.add(user_id)
                unique_items.add(item_id)
            except ValueError:
                continue # Skip malformed lines

    original_user_ids = sorted(list(unique_users))
    original_item_ids = sorted(list(unique_items))

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(original_user_ids)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(original_item_ids)}

    rows, cols, data = [], [], []
    for user_id, item_id, rating in all_ratings:
        if user_id in user_id_to_idx and item_id in item_id_to_idx:
            rows.append(user_id_to_idx[user_id])
            cols.append(item_id_to_idx[item_id])
            data.append(rating)

    num_users = len(original_user_ids)
    num_items = len(original_item_ids)
    full_rating_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    print(f"Full data loaded: {num_users} users, {num_items} items, {full_rating_matrix.nnz} ratings.")
    return full_rating_matrix, user_id_to_idx, item_id_to_idx

def subsample_matrix_from_full(
    full_matrix,
    full_user_to_idx: dict,
    full_item_to_idx: dict,
    n_target: int,
    m_target: int,
    random_seed
) :
    """
    Subsamples a smaller rating matrix from a larger pre-loaded matrix.

    Args:
        full_matrix (csr_matrix): The full sparse rating matrix.
        full_user_to_idx (dict): Original user ID to full matrix index mapping.
        full_item_to_idx (dict): Original item ID to full matrix index mapping.
        n_target (int): Number of users to sample.
        m_target (int): Number of items to sample.
        random_seed (int, optional): Seed for random number generation.

    Returns:
        csr_matrix: The subsampled sparse rating matrix.

    Raises:
        ValueError: If n_target or m_target are larger than available unique users/items.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    original_users = list(full_user_to_idx.keys())
    original_items = list(full_item_to_idx.keys())

    if n_target > len(original_users):
        print(f"Warning: n_target ({n_target}) is greater than available users ({len(original_users)}). Using all available users.")
        selected_user_ids = original_users
    else:
        selected_user_ids = np.random.choice(original_users, n_target, replace=False)
    
    # Map selected original user IDs to their indices in the full_matrix
    selected_user_full_indices = [full_user_to_idx[uid] for uid in selected_user_ids]

    # Create a sub-matrix for the selected users
    # This is efficient for CSR matrices: slicing by row indices
    user_sub_matrix = full_matrix[selected_user_full_indices, :]

    # Now, find items rated by these selected users and choose top M
    item_review_counts = collections.defaultdict(int)
    # Iterate over non-zero elements in the user_sub_matrix
    for _, col_idx in zip(*user_sub_matrix.nonzero()):
        original_item_id = original_items[col_idx] # Get original item ID from its full_matrix index
        item_review_counts[original_item_id] += 1

    unique_items_rated_by_sample = list(item_review_counts.keys())

    if m_target > len(unique_items_rated_by_sample):
        print(f"Warning: m_target ({m_target}) is greater than unique items rated by sampled users ({len(unique_items_rated_by_sample)}). Using all available items rated by sampled users.")
        selected_item_ids = unique_items_rated_by_sample
    else:
        sorted_items_by_review_count = sorted(
            item_review_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
        selected_item_ids = [item_id for item_id, _ in sorted_items_by_review_count[:m_target]]

    # Map selected original item IDs to their indices in the full_matrix
    selected_item_full_indices = [full_item_to_idx[iid] for iid in selected_item_ids]

    # Create a new mapping for the subsampled matrix
    new_user_to_idx = {user_id: idx for idx, user_id in enumerate(selected_user_ids)}
    new_item_to_idx = {item_id: idx for idx, item_id in enumerate(selected_item_ids)}

    rows, cols, data = [], [], []
    for r_idx, original_user_full_idx in enumerate(selected_user_full_indices):
        # Get the row from the full matrix
        row_data = full_matrix.data[full_matrix.indptr[original_user_full_idx]:full_matrix.indptr[original_user_full_idx+1]]
        row_cols = full_matrix.indices[full_matrix.indptr[original_user_full_idx]:full_matrix.indptr[original_user_full_idx+1]]

        for c_val, original_item_full_idx in zip(row_data, row_cols):
            original_item_id = original_items[original_item_full_idx]
            if original_item_id in new_item_to_idx:
                rows.append(new_user_to_idx[original_users[original_user_full_idx]]) # Use new user index
                cols.append(new_item_to_idx[original_item_id]) # Use new item index
                data.append(c_val)

    subsampled_matrix = csr_matrix((data, (rows, cols)), shape=(len(selected_user_ids), len(selected_item_ids)))
    
    return subsampled_matrix

def run_experiments(
    data_path: str,
    n_values,
    m_values,
    d_values,
    num_runs_per_config=1,
    seed=42
): 
    """
    Runs experiments to calculate theoretical and empirical distortion for various
    combinations of n, m, and d.

    Args:
        data_path (str): Path to the MovieLens dataset file (e.g., "u.data").
        n_values (List[int]): List of N (number of users) values to test.
        m_values (List[int]): List of M (number of items) values to test.
        d_values (List[int]): List of D (embedding dimension) values to test.
        num_instances_per_combo (int): Number of instances to generate for each combination.
        base_n (int): Base N value for fixed variables.
        base_m (int): Base M value for fixed variables.
        base_d (int): Base D value for fixed variables.

    Returns:
        pd.DataFrame: A DataFrame containing results for all instances.
    """
    results = []
    total_combinations = len(n_values) * len(m_values) * len(d_values)
    current_combo_idx = 0

    full_rating_matrix, full_user_to_idx, full_item_to_idx = load_full_movielens_data(data_path)


    # Create a list of all combinations to iterate through
    all_combinations = []
    for n in n_values:
        for m in m_values:
            for d in d_values:
                if d >= min(n, m):
                        print(f"Skipping d={d} for n={n}, m={m} as d must be < min(n, m).")
                        continue
                else:
                    all_combinations.append((n, m, d))

    print(f"Starting experiments for {len(all_combinations)} combinations, {num_runs_per_config} instances each.")

    for n, m, d in all_combinations:
        current_combo_idx += 1
        print(f"\n--- Running combo {current_combo_idx}/{total_combinations}: n={n}, m={m}, d={d} ---")

        for i in range(num_runs_per_config):
            print(f"  Instance {i+1}/{num_runs_per_config} for (n={n}, m={m}, d={d})")
            try:
                # Load and sample MovieLens data
                # Using a fixed random seed for sampling within each instance for reproducibility
                # but varying the seed for different instances.
                starting_time= time.time()
                
                '''rating_matrix, user_to_idx, item_to_idx, sparsity = load_and_sample_movielens_data(
                    data_path=data_path,
                    n_to_sample=n,
                    m_to_sample=m,
                    random_seed=i # Use instance index as seed for data sampling
                )'''
                rating_matrix = subsample_matrix_from_full(
                    full_matrix=full_rating_matrix,
                    full_user_to_idx=full_user_to_idx,
                    full_item_to_idx=full_item_to_idx,
                    n_target=n,
                    m_target=m,
                    random_seed=seed+i # Use instance index as seed for subsampling
                )

                # Ensure the matrix is not empty and has valid dimensions for SVD
                if rating_matrix.shape[0] < d or rating_matrix.shape[1] < d or rating_matrix.nnz == 0:
                    print(f"    Skipping instance due to insufficient data for SVD (shape: {rating_matrix.shape}, nnz: {rating_matrix.nnz}).")
                    continue
                
                # Create an Instance object
                instance = Instance(n=rating_matrix.shape[0], m=rating_matrix.shape[1], d=d, noisy_matrix=rating_matrix)
                loading_and_processing_time=time.time()-starting_time
                print("we finished loading and preprocessing in",loading_and_processing_time)
                # Calculate distortions
                starting_time=time.time()
                instance.distortion_comparisons()
                rule_time=time.time()-starting_time
                print("we finished calculating all the distortions in",rule_time)


            except Exception as e:
                print(f"    Error processing instance (n={n}, m={m}, d={d}, instance={i+1}): {e}")
                # Optionally, log the error and continue to the next instance
                continue
    
    return pd.DataFrame(results)

if __name__ == "__main__":
        # 1. Create the parser
    parser = argparse.ArgumentParser(description="Run MovieLens Distortion Experiments.")

    # 2. Add arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default='../data/movie_rating.csv',
        help="Path to the MovieLens u.data (or similar CSV) file."
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default="results/0.csv",
        help="Path to save the experiment results CSV."
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs='*', # 0 or more integers
        default=[10],
        help="List of 'n' (number of users) values to test. Provide as space-separated integers (e.g., 5 10 15)."
    )
    parser.add_argument(
        "--m_values",
        type=int,
        nargs='*',
        default=[50],
        help="List of 'm' (number of items) values to test. Provide as space-separated integers (e.g., 5 10 15)."
    )
    parser.add_argument(
        "--d_values",
        type=int,
        nargs='*',
        default=[2],
        help="List of 'd' (latent dimension) values to test. Provide as space-separated integers (e.g., 2 3 4)."
    )
    parser.add_argument(
        "--num_runs_per_config",
        type=int,
        default=1,
        help="Number of times to repeat each configuration for averaging."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0, # By default, let the function handle it or set to None
        help="Seed for random number generation for reproducibility. If not provided, a new seed might be used per run if your 'num_runs_per_config' logic handles it."
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Use the arguments
    # The lists parsed by argparse will be actual lists of integers.
    # No need to remove duplicates or sort if you're specifying them precisely via CLI.
    # If you still want to ensure unique sorted lists:
    n_values = sorted(list(set(args.n_values)))
    m_values = sorted(list(set(args.m_values)))
    d_values = sorted(list(set(args.d_values)))

    print(f"Running experiments with n_values: {n_values}")
    print(f"Running experiments with m_values: {m_values}")
    print(f"Running experiments with d_values: {d_values}")
    print(f"Number of runs per config: {args.num_runs_per_config}")
    print(f"Random seed for this run: {args.random_seed}")


    # Run the experiments
    # Pass the random_seed argument to your load_and_sample_movielens_data function
    # You might need to adjust your run_experiments function to accept and pass this seed
    # through to the sampling function.
    experiment_results_df = run_experiments(
        data_path=args.data_path,
        n_values=args.n_values,
        m_values=args.m_values,
        d_values=args.d_values,
        num_runs_per_config=args.num_runs_per_config,
        seed=args.random_seed # Pass the seed
    )
