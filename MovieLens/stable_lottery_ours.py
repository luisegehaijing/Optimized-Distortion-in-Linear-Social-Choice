import numpy as np
from itertools import combinations
from scipy.optimize import linprog

import numpy as np
from itertools import combinations
from scipy.optimize import linprog

def get_SaW_count(alternative_a_idx, committee_W_indices, utility_matrix):
    """
    Computes |S_a(W)|, the count of voters who prefer alternative 'a' to all alternatives in committee 'W'.

    S_a(W) = {v in V: a >_v W}, where 'a >_v W' means voter 'v' prefers alternative 'a'
    to every alternative in committee 'W'. This is determined by comparing utilities.

    Args:
        alternative_a_idx (int): Index of alternative 'a'.
        committee_W_indices (tuple or list): Indices of alternatives in committee 'W'.
        utility_matrix (np.array): An n x m utility matrix where n is the number of voters
                                   and m is the number of alternatives.
                                   utility_matrix[j, i] is voter j's utility for alternative i.

    Returns:
        int: The count of voters in S_a(W).
    """
    # Get the number of voters from the utility matrix (now rows)
    n_voters = utility_matrix.shape[0]
    count = 0

    # Iterate through each voter
    for v_idx in range(n_voters):
        prefers_a_to_all_in_W = True
        # Check if voter 'v_idx' prefers 'a' to every alternative in 'W'
        # Access: utility_matrix[voter_idx, alternative_idx]
        for w_idx in committee_W_indices:
            # If utility of 'a' is not strictly greater than utility of 'w',
            # then 'a' is not preferred to 'w' by this voter.
            if utility_matrix[v_idx, alternative_a_idx] <= utility_matrix[v_idx, w_idx]:
                prefers_a_to_all_in_W = False
                break # No need to check other alternatives in W for this voter
        # If 'a' is preferred to all alternatives in 'W' by this voter, increment count
        if prefers_a_to_all_in_W:
            count += 1
    return count

def find_stable_lottery(utility_matrix, k):
    """
    Attempts to compute a stable lottery distribution cW over committees of size k.
    A distribution cW is stable if for all alternatives 'a',
    E_{W ~ cW} [|S_a(W)|] <= n/k, where n is the number of voters.

    This function formulates the problem as a linear program and uses scipy.optimize.linprog
    to find a feasible solution for the probabilities of each committee.

    Args:
        utility_matrix (list of lists or np.array): An n x m utility matrix.
                                                  Rows are voters, columns are alternatives.
        k (int): The desired size of the committee.

    Returns:
        tuple: A tuple containing:
               - dict or None: A dictionary representing the stable lottery distribution
                               {committee_tuple: probability} if found, otherwise None.
               - str: A message indicating success or failure.
    """
    # Convert utility_matrix to a numpy array for efficient indexing
    utility_matrix = np.array(utility_matrix)
    n_voters = utility_matrix.shape[0]       # Number of voters (now rows)
    m_alternatives = utility_matrix.shape[1] # Number of alternatives (now columns)
    alternatives = list(range(m_alternatives)) # List of alternative indices

    # 1. Generate all possible committees of size k
    # Each committee is represented as a tuple of alternative indices.
    all_possible_committees = list(combinations(alternatives, k))
    num_committees = len(all_possible_committees)

    
    # 2. Calculate |S_a(W)| for all alternatives 'a' and all possible committees 'W'.
    # This will form the coefficients for the inequality constraints in the LP.
    # SaW_counts[a_idx, w_idx] will store |S_a(W)| for alternative 'a_idx' and committee 'w_idx'.
    SaW_counts = np.zeros((m_alternatives, num_committees))
    for a_idx in alternatives:
        for w_idx, committee_W_tuple in enumerate(all_possible_committees):
            SaW_counts[a_idx, w_idx] = get_SaW_count(a_idx, committee_W_tuple, utility_matrix)
    

    # 3. Set up the Linear Program
    # We are looking for probabilities p_W for each committee W, such that:
    # - p_W >= 0 (probabilities are non-negative)
    # - sum(p_W for all W) = 1 (probabilities sum to 1)
    # - sum(p_W * |S_a(W)| for all W) <= n/k for each alternative 'a' (stability condition)

    # Objective function: Minimize 0*p_W. We are only interested in finding a feasible solution.
    c = np.zeros(num_committees)

    # Inequality constraints (A_ub @ x <= b_ub):
    # Each row in A_ub corresponds to an alternative 'a'.
    # The coefficients are |S_a(W)| for each committee W.
    A_ub = SaW_counts
    # The right-hand side for each inequality is n/k.
    b_ub = np.full(m_alternatives, n_voters / k)

    # Equality constraint (A_eq @ x == b_eq):
    # Sum of all probabilities p_W must be 1.
    A_eq = np.ones((1, num_committees))
    b_eq = np.array([1.0])

    # Bounds for each variable (probability p_W): 0 <= p_W <= 1
    bounds = [(0, 1) for _ in range(num_committees)]

    # Solve the linear program using the 'highs' method (generally robust and fast)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        # If a solution is found, construct the stable lottery distribution dictionary
        stable_lottery_distribution = {}
        for i, prob in enumerate(res.x):
            # Filter out very small probabilities due to floating point precision
            if prob > 1e-9:
                stable_lottery_distribution[all_possible_committees[i]] = prob
        return stable_lottery_distribution, "Stable lottery found successfully."
    else:
        # If no solution is found, return None and the solver's message
        return None, f"Could not find a stable lottery: {res.message}"

def check_stable_lottery(utility_matrix, k, cW_distribution):
    """
    Checks if a given distribution cW over committees of size k is a stable lottery.

    Args:
        utility_matrix (list of lists or np.array): n x m utility matrix.
                                                  Rows are voters, columns are alternatives.
        k (int): Size of the committee.
        cW_distribution (dict): A dictionary where keys are tuples of alternative indices
                                (representing committees) and values are their probabilities.
                                Example: { (0, 1): 0.5, (0, 2): 0.5 }

    Returns:
        tuple: A tuple containing:
               - bool: True if cW_distribution is a stable lottery, False otherwise.
               - dict: A dictionary showing the expected values for each alternative,
                       and whether the condition is met.
    """
    utility_matrix = np.array(utility_matrix)
    n_voters = utility_matrix.shape[0]
    m_alternatives = utility_matrix.shape[1]
    alternatives = list(range(m_alternatives))

    # Calculate the threshold for the stability condition
    threshold = n_voters / k

    results = {}
    is_stable = True

    # Iterate through each alternative 'a' to check the condition
    for a_idx in alternatives:
        expected_SaW = 0.0
        # Calculate the expected value E_{W ~ cW} [|S_a(W)|]
        for committee_W_tuple, prob_W in cW_distribution.items():
            # Basic validation for the committee in the distribution
            if len(committee_W_tuple) != k or len(set(committee_W_tuple)) != k:
                print(f"Warning: Committee {committee_W_tuple} in distribution does not have size k={k} or has duplicate alternatives. This committee will be skipped in calculation.")
                continue

            s_a_w_count = get_SaW_count(a_idx, committee_W_tuple, utility_matrix)
            expected_SaW += prob_W * s_a_w_count

        # Check if the condition is met for the current alternative 'a'
        condition_met = expected_SaW <= threshold + 1e-9 # Add a small epsilon for floating point comparisons
        results[f"Alternative {a_idx}"] = {
            "expected_SaW": expected_SaW,
            "threshold": threshold,
            "condition_met": condition_met
        }
        if not condition_met:
            is_stable = False # If any alternative fails the condition, the lottery is not stable

    return is_stable, results

def compute_lslr_candidate_distribution(stable_committee_lottery, m_alternatives, sqrt_d):
    """
    Computes the Linear Stable Lottery Rule (LSLR) probability distribution
    over individual candidates (alternatives).

    P(c) = (1 / (2 * sqrt(d))) * Pr_{W ~ cW}[c in W]
    where d is the number of alternatives.

    Args:
        stable_committee_lottery (dict): A dictionary representing the stable lottery
                                         distribution over committees, as returned by
                                         find_stable_lottery.
                                         Example: { (0, 1): 0.5, (0, 2): 0.5 }
        m_alternatives (int): The total number of alternatives (d in the formula).

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary representing the LSLR candidate distribution
                       {candidate_idx: probability}.
               - str: A message indicating success or failure.
    """

    lslr_candidate_distribution = {alt_idx: 0.0 for alt_idx in range(m_alternatives)}

    # Calculate Pr_{W ~ cW}[c in W] for each candidate c
    for committee_tuple, prob_W in stable_committee_lottery.items():
        for candidate_idx in committee_tuple:
            if candidate_idx < m_alternatives: # Ensure candidate index is valid
                lslr_candidate_distribution[candidate_idx] += prob_W
            else:
                print(f"Warning: Candidate {candidate_idx} in committee {committee_tuple} is out of bounds for {m_alternatives} alternatives. Skipping.")


    # Apply the 1/(2*sqrt(d)) factor
    for candidate_idx in lslr_candidate_distribution:
        lslr_candidate_distribution[candidate_idx] /= (2 * sqrt_d)

    return lslr_candidate_distribution, "LSLR candidate distribution computed successfully."


# --- Example Usage ---
if __name__ == "__main__":
    # Example Utility Matrix: m=4 alternatives, n=3 voters
    # Rows are alternatives (0, 1, 2, 3), Columns are voters (0, 1, 2)
    # utility_matrix[alt_idx, voter_idx]
    example_utility_matrix = [
        [10, 5, 8],  
        [ 8, 9, 6],  
        [ 6, 7, 10], # Alternative 2 utilities
        [ 5, 6, 7]   # Alternative 3 utilities
    ]

    m_alternatives_example = 3
    
    
    k_committee_size = 2
    candidates= [[0,1],[0,1],[0,1]]
    output=np.zeros(2)

    
    print(f"--- Attempting to find a stable lottery for k={k_committee_size} ---")
    found_lottery_committees, message_committees = find_stable_lottery(example_utility_matrix, k_committee_size)

    if found_lottery_committees:
        print("\nFound Stable Lottery Distribution over Committees:")
        for committee, prob in found_lottery_committees.items():
            print(f"  Committee {committee}: Probability {prob:.4f}")

        print(f"\n--- Verifying the found committee lottery ---")
        is_stable_committees, verification_results_committees = check_stable_lottery(example_utility_matrix, k_committee_size, found_lottery_committees)
        print(f"Is the found committee lottery stable? {is_stable_committees}")
        #for alt, res in verification_results_committees.items():
        #    print(f"  {alt}: Expected |S_a(W)| = {res['expected_SaW']:.4f}, Threshold = {res['threshold']:.4f}, Condition Met: {res['condition_met']}")

        print(f"\n--- Computing LSLR Candidate Distribution ---")
        lslr_candidates, lslr_message = compute_lslr_candidate_distribution(found_lottery_committees, m_alternatives_example, k_committee_size)
        if lslr_candidates:
            print("\nLSLR Candidate Distribution:")
            for candidate_idx, prob in lslr_candidates.items():
                print(f"  Candidate {candidate_idx}: Probability {prob:.4f}")
                print("candidate_idx",candidate_idx)
                output+=prob*np.array(candidates[candidate_idx])
            print(f"  Sum of candidate probabilities: {sum(lslr_candidates.values()):.4f}")
        output+=0.5*np.array([1/2,1/2])

    print(output)
        


    