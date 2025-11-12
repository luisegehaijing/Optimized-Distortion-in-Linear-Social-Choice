import pandas as pd
import numpy as np
import io
import os 
import time
import instance_embedding
from instance_embedding import Instance

global opinions
opinions = [
            "I believe that abortion should be legal and accessible",
            "I think abortion should be a personal decision between a woman and her doctor",
            "I believe that abortion should be allowed in the first trimester but restricted afterward ",
            "I think abortion should be restricted to certain circumstances",
            "I believe that abortion should be illegal because it involves taking a human life, which I consider "
        ]

def convert_csv_to_choice_matrix(csv_filepath: str) -> pd.DataFrame:
    """
    Converts a CSV file into a matrix where each row represents a user_id
    and columns correspond to specific opinions, filled with the 'choice_numeric'
    value for that opinion. The order of opinions in the input CSV is not assumed.

    Args:
        csv_filepath (str): The file path to the CSV data.

    Returns:
        pd.DataFrame: A DataFrame representing the 100x5 matrix, where columns
                      are ordered by the predefined opinions.
                      Returns an empty DataFrame if no data or issues are found.
    """
    try:
        # Check if the file exists
        if not os.path.exists(csv_filepath):
            print(f"Error: CSV file not found at '{csv_filepath}'")
            return pd.DataFrame()

        # Read the CSV data from the file path
        df = pd.read_csv(csv_filepath)

        # Ensure required columns exist
        required_columns = ['user_id', 'question_type', 'choice_numeric']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing one or more required columns. Please ensure your CSV has: {required_columns}")
            return pd.DataFrame()

        # Define the specific opinions and their corresponding column names
        # The order here defines the column order in the output matrix.
        # We will map the 'question_type' text from the CSV to these predefined opinions.
      
        # Create a mapping from opinion text to its desired column index
        opinion_to_col_index = {opinion: i for i, opinion in enumerate(opinions)}
        column_names = [f'Opinion_{i+1}' for i in range(len(opinions))] # e.g., Opinion_1, Opinion_2, etc.

        # Prepare data for the matrix
        matrix_data = []
        unique_user_ids = df['user_id'].unique()

        # Sort user_ids to ensure consistent row order in the output matrix
        unique_user_ids.sort()

        for user_id in unique_user_ids:
            if user_id.startswith("generation"):
                user_rows = df[df['user_id'] == user_id]
                
                # Initialize a row for the current user with NaN values
                # This ensures that if an opinion is missing for a user, its entry is NaN
                user_matrix_row = [np.nan] * len(opinions)
                
                found_opinions_count = 0
                for _, row in user_rows.iterrows():
                    question_text = row['question_text']
                    choice_numeric_value = row['choice_numeric']
                   # print("question_text",question_text)

                    matched_opinion = False
                    for col_idx, opinion_text in enumerate(opinions):
                        if opinion_text in question_text:
                            user_matrix_row[col_idx] = choice_numeric_value
                            found_opinions_count += 1
                            matched_opinion = True
                            break # Assume one que
                # else:
                #     print(f"Warning: User ID {user_id} has an unrecognized question_type: '{question_type_text}'. Skipping this entry.")

                # Only add the user's row if they have exactly 5 opinions recorded
                if found_opinions_count == len(opinions):
                    matrix_data.append(user_matrix_row)
                else:
                    print(f"Warning: User ID {user_id} does not have exactly {len(opinions)} expected opinion entries. Found: {found_opinions_count}. Skipping this user.")

        # Create the DataFrame for the matrix
        result_matrix = pd.DataFrame(matrix_data, columns=column_names, index=unique_user_ids[:len(matrix_data)])

        # Ensure the matrix has 100 rows if that's an explicit requirement
        if len(result_matrix) != 100:
            print(f"Warning: The generated matrix has {len(result_matrix)} rows, not 100. This might be due to fewer than 100 unique users or users not meeting the 5-opinion criteria.")
            # You might want to pad with empty rows or raise an error if 100 rows is strict.

        return result_matrix

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def run_experiments(
    matrix, 
    n=100,
    m=5,
    d_values=[],
    num_runs_per_config=1,
    seed=42
): 
   
    results = []
    current_combo_idx = 0

    print(f"Starting experiments for {len(d_values)} combinations, {num_runs_per_config} instances each.")

    for d in d_values:
        current_combo_idx += 1
        print(f"\n--- Running combo {current_combo_idx}/{len(d_values)}: n={n}, m={m}, d={d} ---")

        for i in range(num_runs_per_config):
            print(f"  Instance {i+1}/{num_runs_per_config} for (n={n}, m={m}, d={d})")
            try:
                # Load and sample MovieLens data
                # Using a fixed random seed for sampling within each instance for reproducibility
                # but varying the seed for different instances.
                starting_time= time.time()
            
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


# --- Example Usage ---
if __name__ == "__main__":
    # Define the filename
    csv_filename = "../data/abortion_survey.csv"
    rating_matrix = convert_csv_to_choice_matrix(csv_filename)

    run_experiments( 
    matrix=rating_matrix,
        d_values=[2,3,4,5]
        #d_values= [128]
    )

   