import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# (Your parse_experiment_log function remains the same)
def parse_experiment_log(log_content,parsed_data=[]):
    """
    Parses the provided experiment log content to extract distortion and running time data
    using string splitting methods. This version correctly handles metrics spread across
    multiple lines for each rule.

    Args:
        log_content (str): A string containing the full log output from the experiments.

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed data with columns for
                          'n', 'rule', 'empirical_distortion', 'theoretical_distortion',
                          'running_time'.
    """

    current_n = None
    current_m = None
    current_d = None
    current_rule = None
    current_metrics_for_rule = {} # Temporary dictionary to hold metrics for the current rule

    lines = log_content.splitlines()

    for line in lines:
        stripped_line = line.strip()

        # Check for new combo (e.g., "--- Running combo 1/6: n=3, m=5, d=2 ---")
        if stripped_line.startswith("--- Running combo"):
            try:
                # Extract the part after "n=", "m=", "d="
                parts = stripped_line.split(":")[-1].strip().split(", ")
                n_str = [p for p in parts if p.startswith("n=")][0].split("=")[1]
                m_str = [p for p in parts if p.startswith("m=")][0].split("=")[1]
                d_str = [p for p in parts if p.startswith("d=")][0].split("=")[1]
                d_str = d_str.split(" ")[0]
                
                current_n = int(n_str)
                current_m = int(m_str)
                current_d = int(d_str)
                # Reset rule and metrics when a new combo starts
                current_rule = None
                current_metrics_for_rule = {}
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse combo line '{stripped_line}'. Error: {e}")
                current_n, current_m, current_d = None, None, None # Reset if parsing fails
            continue

        # Check for rule name (e.g., "Rule: Plurality")
        if stripped_line.startswith("Rule:"):
            # If we were previously collecting metrics for a rule, and a new rule starts,
            # it means the previous rule's data might be incomplete or not correctly added.
            # For this specific log structure, it implies we should have finished the previous rule.
            # If not all metrics were found, we'll just move on.
            current_rule = stripped_line.split("Rule:")[1].strip()
            current_metrics_for_rule = {} # Reset for the new rule
            continue

        # If we have a current rule and n value, try to extract its metrics
        if current_rule and current_n is not None:
            if "Empirical Distortion:" in stripped_line:
                try:
                    current_metrics_for_rule['empirical_distortion'] = float(stripped_line.split("Empirical Distortion:")[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse empirical distortion from '{stripped_line}'")
            elif "Theoretical Distortion:" in stripped_line:
                try:
                    current_metrics_for_rule['theoretical_distortion'] = float(stripped_line.split("Theoretical Distortion:")[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse theoretical distortion from '{stripped_line}'")
            elif "Running Time:" in stripped_line:
                try:
                    current_metrics_for_rule['running_time'] = float(stripped_line.split("Running Time:")[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse running time from '{stripped_line}'")

            # Check if all three core metrics are collected for the current rule
            # The 'Theoretical Bound Calculating Time' and 'Iterations' lines are ignored for this.
            if all(k in current_metrics_for_rule for k in ['empirical_distortion','running_time','theoretical_distortion']):
                parsed_data.append({
                    'n': current_n,
                    'm': current_m,
                    'd': current_d,
                    'rule': current_rule,
                    'empirical_distortion': current_metrics_for_rule['empirical_distortion'],
                    'theoretical_distortion': current_metrics_for_rule['theoretical_distortion'],
                    'running_time': current_metrics_for_rule['running_time']
                })
                current_rule = None # Reset rule after processing its metrics
                current_metrics_for_rule = {} # Clear collected metrics for the next rule
    return parsed_data

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration for file reading ---
    log_file_path_1 = 'abortion_d.1098478' # <--- IMPORTANT: Change this to your actual log file path

    try:
        with open(log_file_path_1, 'r') as f:
            log_data_1 = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path_1}' was not found.")
        print("Please make sure your experiment results are saved in a text file")
        print("and update the 'log_file_path' variable in the script.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit()

    df=pd.DataFrame(parse_experiment_log(log_data_1,[]))

    if df.empty:
        print("No data was parsed. Please check your log file format.")
        exit()

    print("Parsed Data Head:")
    print(df.head())
    print("\nParsed Data Info:")
    df.info()

    # --- Plotting the results ---
    

    rule_name_map = {
        'uniform': 'UProj/Uniform',
        'Plurality': 'Plurality/MCP/Borda',
        'Random Dictatorship': 'Random Dictatorship',
        'Random Harm': 'Random Harmonic',
        'Randomized Instance Optimal': 'Randomized Optimal',
        'Deterministic Instance Optimal': 'Deterministic Optimal',
        'Linear Stable Lottery Rule (LSLR)': 'LSLR'
    }
    df['rule_display_name'] = df['rule'].map(rule_name_map).fillna(df['rule'])

    # Define your rule categories (Deterministic vs. Randomized)
    deterministic_rules = [
        
        'Plurality/MCP/Borda',
        'Deterministic Optimal',
        
    ]
    randomized_rules = [
        'Random Dictatorship',
        'Random Harmonic',
        'UProj/Uniform',
        'LSLR',
        'Randomized Optimal'
    ]

    # Create a new column 'rule_type'
    df['rule_type'] = df['rule_display_name'].apply(
        lambda x: 'Deterministic' if x in deterministic_rules else (
            'Randomized' if x in randomized_rules else 'Other' # Handle any rules not explicitly categorized
        )
    )

    custom_rule_order = [
        
        'Plurality/MCP/Borda',
        'Deterministic Optimal',
        'Random Dictatorship',
        'Random Harmonic',
        'UProj/Uniform',
        'LSLR',
        'Randomized Optimal'
    ]

    line_style_map = {
        'Deterministic': (None, None), # Solid line (no dashes)
        'Randomized': (4, 4),       # Dashed line (4 points on, 4 points off)
        'Other': (1, 1)             # Dotted line (1 point on, 1 point off)
    }

    plt.figure(figsize=(10, 7))
    ax=sns.lineplot(
        data=df,
        x='d',
        y='empirical_distortion',
        hue='rule_display_name',
        hue_order=custom_rule_order,
        style='rule_type', # Use the new 'rule_type' column for line styles
        dashes=line_style_map, # Map rule types to specific line styles
        marker="o",
        markersize=7,
        alpha=1,
        linewidth=2,
        palette="deep"
    )
    sns.despine()
    
    #plt.title('Abortion Survey Empirical Distortion (m=5, n=100)', fontsize=20)
    plt.ylim(0.98, 1.5)
    plt.xlabel('Number of Dimensions (d)', fontsize=20)
    plt.ylabel('Distortion', fontsize=20)
    plt.xticks(df['d'].unique())
    ax.tick_params(axis='both', labelsize=20)
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    leg_lines = leg.get_lines()
    leg_lines[3].set_linestyle(":")
    leg_lines[4].set_linestyle(":")
    leg_lines[5].set_linestyle(":")
    leg_lines[6].set_linestyle(":")
    leg_lines[7].set_linestyle(":")
    #plt.legend(title='Rule', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.savefig('d_emprical_distortion.png')

    plt.figure(figsize=(10, 7))
    ax=sns.lineplot(
        data=df,
        x='d',
        y='theoretical_distortion',
        hue='rule_display_name',
        hue_order=custom_rule_order,
        style='rule_type', # Use the new 'rule_type' column for line styles
        dashes=line_style_map, # Map rule types to specific line styles
        marker="o",
        markersize=7,
        alpha=1,
        linewidth=2,
        palette="deep"
    )
    #plt.title('Abortion Survey Distortion Upper Bound (m=5, n=100)', fontsize=20)
    plt.ylim(1.1, 1.7)
    sns.despine()
    plt.xlabel('Number of Dimensions (d)', fontsize=20)
    plt.ylabel('Distortion ', fontsize=20)
    plt.xticks(df['d'].unique())
    ax.tick_params(axis='both', labelsize=20)
    #plt.legend(title='Rule', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    leg_lines = leg.get_lines()
    leg_lines[3].set_linestyle(":")
    leg_lines[4].set_linestyle(":")
    leg_lines[5].set_linestyle(":")
    leg_lines[6].set_linestyle(":")
    leg_lines[7].set_linestyle(":")
    plt.tight_layout()
    plt.savefig('d_theoretical_distortion.png')

    plt.figure(figsize=(10, 7))
    ax=sns.lineplot(
        data=df,
        x='d',
        y='running_time',
        hue='rule_display_name',
        hue_order=custom_rule_order,
        style='rule_type', # Use the new 'rule_type' column for line styles
        dashes=line_style_map, # Map rule types to specific line styles
        marker="o",
        markersize=7,
        alpha=1,
        linewidth=2,
        palette="deep"
    )
    plt.title('Abortion Survey Running Time (m=5, n=100) ', fontsize=16)
    plt.xlabel('Number of Dimensions (d)', fontsize=20)
    plt.ylabel('Running Time (seconds)', fontsize=20)
    plt.xticks(df['d'].unique())
    sns.despine()
    ax.tick_params(axis='both', labelsize=20)
    #plt.legend(title='Rule', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    leg_lines = leg.get_lines()
    leg_lines[3].set_linestyle(":")
    leg_lines[4].set_linestyle(":")
    leg_lines[5].set_linestyle(":")
    leg_lines[6].set_linestyle(":")
    leg_lines[7].set_linestyle(":")
    plt.tight_layout()
    plt.savefig('d_running_time.png')