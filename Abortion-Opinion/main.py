import numpy as np
import collections
from scipy.sparse import csr_matrix
from scipy.optimize import minimize,linprog
from scipy.sparse.linalg import svds
import gurobipy as gp
from sklearn.decomposition import PCA
from stable_lottery import find_stable_lottery, check_stable_lottery,compute_lslr_candidate_distribution,get_SaW_count
from typing import Optional, Tuple, List
import time
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings



opinions = [
            "I believe that abortion should be legal and accessible because women have the right to make decisions about their own bodies. Access to safe abortions is crucial for protecting women’s health and well-being. Society should support comprehensive sex education and contraception to reduce the need for abortions.",
            "I think abortion should be a personal decision between a woman and her doctor, without government interference. Each situation is unique, and women should have the autonomy to make the best choice for themselves and their families. Society should ensure that all women have access to affordable healthcare, including reproductive services.",
            "I believe that abortion should be allowed in the first trimester but restricted afterward unless there are exceptional circumstances. This policy respects a woman’s right to choose while recognizing the increasing moral considerations as the pregnancy progresses. Society should invest in education and healthcare to prevent unwanted pregnancies and support women through their reproductive choices.",
            "I think abortion should be restricted to certain circumstances, such as cases of rape, incest, or when the mother’s life is at risk. This approach balances the rights of the unborn with the needs of women facing difficult situations. Society should provide support for women who carry their pregnancies to term, including healthcare and financial assistance.",
            "I believe that abortion should be illegal because it involves taking a human life, which I consider morally wrong. Society should focus on providing resources and support for pregnant women to encourage them to choose life. Adoption should be promoted as a viable alternative to abortion."
        ]

def is_uniform_vector_in_convex_hull(d: int, vectors):
   
    
    target_vector = np.full(d, 1.0 / d)
    np_vectors = np.array(vectors)

    
    # 'num_vectors' is the count of the input vectors provided.
    num_vectors = np_vectors.shape[0]
    c = np.zeros(num_vectors)
    A_eq = np.zeros((d + 1, num_vectors))
    b_eq = np.zeros(d + 1)

    for dim_idx in range(d):
        A_eq[dim_idx, :] = np_vectors[:, dim_idx]
        b_eq[dim_idx] = target_vector[dim_idx]
   
    A_eq[d, :] = 1.0
    b_eq[d] = 1.0
    

    bounds = [(0.0, None)] * num_vectors
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(f"\n--- linprog Result for d={d}, vectors={vectors} ---")
    print(f"Full result object: {res}")
    print(f"res.success: {res.success}")
    print(f"res.fun: {res.fun}")
    print(f"res.status: {res.status}")
    print(f"res.message: {res.message}")
    print("--------------------------------------------------")
    # --- End Debugging Prints ---

    # The target vector is in the convex hull if:
    # 1. The optimization was successful (meaning a feasible solution was found).
    # 2. The objective function value is approximately zero (since our 'c' vector was all zeros).
    
    return res.success and (np.isclose(res.fun, 0.0)).all()


def language_embeddings(opinions,d):
    model_name = "tomaarsen/mpnet-base-nli-matryoshka" # A good choice specifically optimized for 64

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    # Define the desired target dimension for truncation
    target_dimension =768 #64

    # 2. Encode the alternatives with the specified truncation dimension
    print(f"Deco alternatives to {d} dimensions...")
    embeddings_mrl = model.encode(opinions)[:, :target_dimension]
    pca = PCA(n_components=d)
    principal_components = pca.fit_transform(embeddings_mrl)

    #return embeddings_mrl
    print("principal_components",principal_components)

   

    return principal_components


def learn_l1_normalized_vector(item_embeddings, utility_vector):
    """
    Learns an L1-normalized vector 'w' such that item_embeddings @ w approximates utility_vector
    for non-zero entries of utility_vector, using non-negative constraints.

    Args:
        item_embeddings (np.ndarray): A (num_items, embedding_dim) array of item embeddings.
        utility_vector (np.ndarray): A (num_items,) array representing a user's utility for items.
                                     Zero entries indicate unknown or irrelevant utilities.

    Returns:
        np.ndarray: The learned L1-normalized (and non-negative) vector 'w', or None if
                    optimization fails.
    """
    num_items, embedding_dim = item_embeddings.shape

    def objective_function(w):
        predicted_utility = item_embeddings @ w
        non_zero_indices = utility_vector != 0
        filtered_predicted_utility = predicted_utility[non_zero_indices]
        filtered_actual_utility = utility_vector[non_zero_indices]

        # Calculate MSE only on the filtered (non-zero) entries
        if len(filtered_actual_utility) > 0:
            mse = np.sum(np.square(filtered_predicted_utility - filtered_actual_utility))
        else:
            mse = 0.0
        return mse

    # Constraint for L1 norm: sum(abs(w_i)) = 1
    def l1_constraint(w):
        return np.sum(np.abs(w)) - 1

    constraints = ({'type': 'eq', 'fun': l1_constraint})
    
    # Bounds for non-negativity: 0 <= w_i <= 1
    bounds = [(0, 1)] * embedding_dim

    # Initial guess for w (random, then L1-normalize)
    w_initial = np.random.rand(embedding_dim)
    sum_w_initial = np.sum(w_initial)
    w_initial = w_initial / sum_w_initial if sum_w_initial != 0 else w_initial

    result = minimize(
        fun=objective_function,
        x0=w_initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        # Re-normalize just in case floating point errors slightly violate L1 norm
        # Ensure non-negativity more strictly after optimization if needed, but bounds should handle it.
        learned_w = result.x
        # Clip negative values that might arise due to floating point inaccuracies near zero
        learned_w[learned_w < 0] = 0
        sum_learned_w = np.sum(learned_w)
        # Handle cases where sum might be zero after clipping (e.g., if all initial values were near zero)
        learned_w = learned_w / sum_learned_w if sum_learned_w != 0 else learned_w
        return learned_w
    else:
        print(f"Optimization failed: {result.message}")
        return None

class Instance:
    def __init__(self, n: int, m: int, d: int, noisy_matrix: csr_matrix):
        """
        Initializes an Instance for distortion analysis.

        Args:
            n (int): Number of voters (users).
            m (int): Number of candidates (items).
            d (int): Dimensionality of the embedding space.
            noisy_matrix (csr_matrix): The sparse rating matrix from which embeddings are derived.
        """
        self.EPS_OBJ_VIOLATION= 1e-7
        self.n = n
        self.m = m
        self.d = d
        self.voters = []  # User embeddings
        self.candidates = []  # Item embeddings
        self.utilities = None  # Full utility matrix (n x m)
        self.c_star = None  # Optimal candidate (embedding) maximizing social welfare
        self.max_social_welfare = 0.0

        self.model: Optional[gp.Model] = None # Gurobi model for feasibility region
        self.v_vars = {} # Gurobi variables for voter embeddings
        self.vbar_vars = {} # Gurobi variables for average voter embedding

        self.generate_instance(noisy_matrix)
        self.warm_start() # Initialize the Gurobi model for theoretical feasibility checks
        print("each candidate's utility is", self.total_utilities_per_candidate)
        
    def generate_instance(self, noisy_matrix: csr_matrix):

        item_embeddings=language_embeddings(opinions,self.d)

        # L1 normalization for item embeddings and ensure positive entries
        min_val_item = np.min(item_embeddings)
        
        # Shift to make all values non-negative before L1 normalization
        item_embeddings_shifted = item_embeddings - min_val_item if min_val_item < 0 else item_embeddings
        item_embeddings_normalized = np.zeros_like(item_embeddings_shifted)

        for i, embedding in enumerate(item_embeddings_shifted):
            l1_norm = np.linalg.norm(embedding, ord=1)
            if l1_norm > 0:
                item_embeddings_normalized[i] = embedding / l1_norm
            else:
                # If L1 norm is zero (all entries are zero), keep it as is or assign a small non-zero value
                # This case is rare if embeddings are meaningful.
                item_embeddings_normalized[i] = embedding # Or a default valid vector if all zeros is problematic

        self.candidates = item_embeddings_normalized

        # Learn L1-normalized user vectors (voters)
        # For each user, their utility vector is a row from the reconstructed_utility matrix.
        # The `learn_l1_normalized_vector` function expects a single user's utility vector.
        learned_user_embeddings = []
        reconstructed_utility = (1/6) * np.array(noisy_matrix)
        for i in range(self.n):
            # Pass the item_embeddings_normalized to the learning function
            # and the i-th row of the reconstructed_utility (which represents user i's utility for all items).
            if reconstructed_utility.shape[0] > i:
                user_util_vector = reconstructed_utility[i, :] 
                learned_v = learn_l1_normalized_vector(self.candidates, user_util_vector)
                if learned_v is not None:
                    learned_user_embeddings.append(learned_v)
                else:
                    # Handle cases where learning fails for a user, e.g., assign a default vector or skip
                    print(f"Warning: Failed to learn L1-normalized vector for user {i}. Skipping or assigning default.")
                    # For now, if it fails, the list will be shorter than self.n.
                    # A robust solution might involve assigning a random L1 vector or raising an error.
            else:
                print(f"Warning: reconstructed_utility has fewer rows than self.n. Missing utility for user {i}.")

        self.voters = np.array(learned_user_embeddings)

        # Re-calculate utilities using the L1-normalized voters and candidates
        # This self.utilities is the "true" utility in the latent space
       
        self.utilities = self.voters @ self.candidates.T

        # Determine the socially optimal candidate (c_star) and its welfare
        self.total_utilities_per_candidate = np.sum(self.utilities, axis=0)
        
        self.c_star = self.candidates[np.argmax(self.total_utilities_per_candidate)]
        self.max_social_welfare = np.max(self.total_utilities_per_candidate)
        print("candidates",self.candidates)
        #print("voters",self.voters)
        #print("utilities",self.utilities)
        print("utilities per candidate",self.total_utilities_per_candidate)
        print("we have finished generating the instnace.")
        print("is the uniform vector inside the convex hull?")
        hey=is_uniform_vector_in_convex_hull(self.d,self.candidates)
        print("hey",hey)

    def warm_start(self):
        """
        Initializes the Gurobi model for the feasibility region of v_k vectors and vbar.
        This model will be used in check_theoretical_feasibility_lp.
        """
        model = gp.Model("RankingConsistencyLP")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Define variables as instance attributes for access in check_theoretical_feasibility_lp
        self.v_vars = model.addVars(self.n, self.d, name="v", lb=0, ub=1)
        self.vbar_vars = model.addVars(self.d, name="vbar", lb=0, ub=1)
        self.alpha_vars = model.addVars(self.n, self.m, name="alpha", lb=0.0, ub=1)

        for k in range(self.n):  # Iterate through each voter
            ##pairwise constraints
            for i in range(self.m):  # Candidate i
                for j in range(self.m):  # Candidate j
                    if i == j:
                        continue  # Skip self-comparison

                    # If voter k prefers candidate i over candidate j
                    if self.utilities[k, i] > self.utilities[k, j]:
                        # Constraint: v_k . (candidate_embeddings[i] - candidate_embeddings[j]) >= 0
                        diff_embedding = self.candidates[i] - self.candidates[j]
                        model.addConstr(
                            gp.quicksum(self.v_vars[k, dim] * diff_embedding[dim] for dim in range(self.d)) >= 0,
                            name=f"voter{k}_pref_c{i}_over_c{j}"
                        )
            ##l1 normalization constraints
            model.addConstr(
                gp.quicksum(self.v_vars[k, dim] for dim in range(self.d)) == 1,
                name=f"voter{k}_l1_norm"
            )

            
            ##cone conditions
            ''' for dim in range(self.d):
                model.addConstr(
                    self.v_vars[k, dim] == gp.quicksum(self.alpha_vars[k, j] * self.candidates[j, dim] for j in range(self.m)),
                    name=f"voter{k}_cone_dim{dim}"
                )

            model.addConstr(
                gp.quicksum(self.alpha_vars[k, j] for j in range(self.m)) == 1,
                name=f"voter{k}_alpha_sum_to_1"
            )'''
            

        # Add constraints to define vbar as the average of v_k vectors
        for dim in range(self.d):
            model.addConstr(
                self.vbar_vars[dim] == (1/self.n) * gp.quicksum(self.v_vars[k, dim] for k in range(self.n)),
                name=f"vbar_avg_dim{dim}"
            )

        model.update()
        self.model = model 

        # --- Initial Feasibility Check ---
        # Set a dummy objective to check feasibility
        self.model.setObjective(0, gp.GRB.MINIMIZE) 
        try:
            self.model.optimize()
            if self.model.status == gp.GRB.INFEASIBLE:
                print(f"  Warning: Gurobi model is INFEASIBLE during warm_start (status: {self.model.status}).")
                # Optionally, compute IIS here for detailed debugging
                # self.model.computeIIS()
                # self.model.write("infeasible_warm_start.ilp")
                self.is_feasible_instance = False
            elif self.model.status == gp.GRB.OPTIMAL:
                print(f"  Gurobi model is OPTIMAL (feasible) during warm_start.")
                self.is_feasible_instance = True
            else:
                print(f"  Warning: Gurobi model has unexpected status during warm_start: {self.model.status}.")
                self.is_feasible_instance = False # Consider other non-optimal statuses as problematic
        except gp.GurobiError as e:
            print(f"  Gurobi error during warm_start initial optimization: {e}")
            self.is_feasible_instance = False

    def check_theoretical_feasibility_lp(
            self,
            beta_test: float,
            c_chosen: np.ndarray,
            c_star,
            start_basis: Optional[Tuple[List[int], List[int]]] = None # Optional: (VBasis, CBasis) for warm start
    ) -> Tuple[bool, Optional[Tuple[int, int]]]: # Changed return type for basis to int, int
        """
        Checks the theoretical feasibility for a given beta_test and chosen candidate.

        Args:
            beta_test (float): The beta value to test for feasibility.
            c_chosen (np.ndarray): The embedding of the chosen candidate.
            start_basis (Optional[Tuple[List[int], List[int]]]): Gurobi basis for warm start.

        Returns:
            Tuple[bool, Optional[Tuple[List[int], List[int]]]]:
                - True if feasible, False otherwise.
                - The Gurobi basis (varbasis, constrbasis) if optimal/feasible, None otherwise.
        """
        if self.model is None:
            raise ValueError("Gurobi model not initialized. Call warm_start() first.")

        model = self.model

        # Define the objective function for minimization
        # Objective: minimize (vbar . c_chosen) - beta_test * (vbar . c_star)
        # Note: self.c_star and c_chosen are numpy arrays.
        vbar_dot_c_star = gp.quicksum(self.vbar_vars[dim] * c_star[dim] for dim in range(self.d))
        vbar_dot_c_chosen = gp.quicksum(self.vbar_vars[dim] * c_chosen[dim] for dim in range(self.d))
        
        model.setObjective(vbar_dot_c_chosen - beta_test * vbar_dot_c_star, gp.GRB.MINIMIZE)

        # Apply warm start basis if provided
        if start_basis:
            vbasis, cbasis = start_basis
            try:
                # Assign VBasis to variables
                for i, var in enumerate(model.getVars()):
                    if i < len(vbasis):
                        var.VBasis = vbasis[i]
                # Assign CBasis to constraints
                for i, constr in enumerate(model.getConstrs()):
                    if i < len(cbasis):
                        constr.CBasis = cbasis[i]
                model.setParam('Method', 0)  # Use primal simplex to leverage basis
            except gp.GurobiError as e:
                print(f"Warning: Error applying warm start basis: {e}")
                model.setParam('Method', -1) # Reset method to auto if basis fails
        else:
            model.setParam('Method', -1) # Auto-select method if no warm start

        # Optimize the model
        model.optimize()
        current_basis = None
        if model.status == gp.GRB.OPTIMAL:
            
            is_feasible_for_beta = model.ObjVal > 0  
            try:
                vbasis = [var.VBasis for var in model.getVars()]
                cbasis = [constr.CBasis for constr in model.getConstrs()]
                current_basis = (vbasis, cbasis)
            except gp.GurobiError as e:
                print(f"Warning: Could not retrieve basis: {e}")
                current_basis = None
            return is_feasible_for_beta, current_basis
        else:
            return False, None
 
    def find_violated_among_candidate_reverse(self, beta_hat: float, c_chosen):
            violated_cuts_data = []
        
            model = self.model
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', '')

            for k_idx, c_k in enumerate(self.candidates):
                objective_vector = beta_hat * c_chosen-c_k
                
                gurobi_obj_expr = gp.quicksum(objective_vector[dim] * self.vbar_vars[dim]
                                            for dim in range(self.d))
                model.setObjective(gurobi_obj_expr, gp.GRB.MINIMIZE)

                try:
                    model.optimize()

                    if model.status == gp.GRB.OPTIMAL:
                        min_obj_val = model.ObjVal
                        v_bar_solution_val = np.array([self.vbar_vars[dim].X for dim in range(self.d)])

                        if min_obj_val < -self.EPS_OBJ_VIOLATION:
                            violated_cuts_data.append((v_bar_solution_val, c_k))
                    
                    elif model.status == gp.GRB.INF_OR_UNBD:
                        print(f"Warning: Gurobi separation oracle for c_k (index {k_idx}) returned INF_OR_UNBD. Status: {model.status}.")
                    else:
                        print(f"Warning: Gurobi separation oracle for c_k (index {k_idx}) returned non-optimal status: {model.status}.")

                except gp.GurobiError as e:
                    print(f"Gurobi Error in find_violated_constraints for c_k (index {k_idx}): {e}. Skipping this candidate.")
                except Exception as e:
                    print(f"An unexpected error occurred during Gurobi optimization for c_k (index {k_idx}): {e}. Skipping this candidate.")
                    
            return violated_cuts_data    

    def calculate_theoretical_distortion_unconstrained(self, c_chosen, max_iter=100, tolerance=1e-6, verbose=False):
        master_theo_model = gp.Model("MasterLP")
        master_theo_model.setParam('OutputFlag', 0)
        master_theo_model.setParam('LogFile', '')
        beta_var = master_theo_model.addVar(name="beta", lb=0.0, ub=1.0)
        master_theo_model.setObjective(beta_var, gp.GRB.MAXIMIZE)
        
        optimal_beta = None  # Initialize optimal_beta
        
        for iteration in range(1, max_iter + 1):
            if verbose:
                print(f"\nIteration {iteration}: Solving Master Problem with {master_theo_model.NumConstrs} constraints...")
            
            try:
                master_theo_model.optimize()
                
                if master_theo_model.status not in [gp.GRB.OPTIMAL, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED]:
                    print(f"Master problem failed to solve or is not optimal at iteration {iteration}: {master_theo_model.status}")
                    return None
                
                # If master model is unbounded or infeasible, it's problematic
                if master_theo_model.status == gp.GRB.INF_OR_UNBD or master_theo_model.status == gp.GRB.UNBOUNDED:
                    print(f"Master problem is {master_theo_model.status} at iteration {iteration}. Returning None.")
                    return None
                
                current_beta_hat = beta_var.X
                if current_beta_hat is None:
                    print(f"Solver returned None values at iteration {iteration}. Status: {master_theo_model.status}")
                    return None
                
                if verbose:
                    print(f"Master solution: beta_hat={current_beta_hat:.6f}")
                
                # Call the separation oracle to find violated constraints
                # This will use the model (the other Gurobi model)
                violated_cuts_data = self.find_violated_among_candidate_reverse(current_beta_hat, c_chosen)
                
                if not violated_cuts_data:
                    if verbose:
                        print("No violated constraints found. Optimal solution reached.")
                    optimal_beta = current_beta_hat
                    break
                else:
                    for v_star, c_k_violated in violated_cuts_data:
                        # The cut is: <c_chosen, v_star> - beta_var * <c_k_violated, v_star> >= 0
                        # This is a linear constraint in beta_var
                        #lhs_constant = np.dot(c_chosen, v_star)
                        #rhs_coeff = np.dot(c_k_violated, v_star)
                        lhs_constant = np.dot(c_k_violated, v_star) #the new cut should be rhs_coeff* beta -<c_k,v> >=0
                        rhs_coeff = np.dot(c_chosen, v_star)
                       
                        # Add the new constraint to the master problem
                        # np.where is used to get the index of c_k_violated for naming the cut
                        # This assumes candidates are unique for reliable indexing.
                        c_k_idx = np.where((self.candidates == c_k_violated).all(axis=1))[0][0]
                       # master_theo_model.addConstr(lhs_constant - beta_var * rhs_coeff >= 0,
                        master_theo_model.addConstr( beta_var * rhs_coeff-lhs_constant >= 0,
                                                name=f"cut_iter{iteration}_ck{c_k_idx}_v{iteration}")
                    
                    master_theo_model.update()
                    
                    if verbose:
                        print(f"Found {len(violated_cuts_data)} new cut(s). Total cuts: {master_theo_model.NumConstrs}")
            
            except gp.GurobiError as e:
                print(f"Gurobi Error in Master Problem at iteration {iteration}: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred at iteration {iteration}: {e}")
                return None
        
        else:
            if verbose:
                print(f"Maximum iterations ({max_iter}) reached without full convergence.")
            # If we exit the loop without finding optimal solution, use the last beta value
            if 'current_beta_hat' in locals():
                optimal_beta = current_beta_hat
            else:
                print("No valid solution found.")
                return None
        
        # Check if optimal_beta is valid before computing theoretical distortion
        if optimal_beta is None or optimal_beta <= 0:
            print(f"Invalid optimal_beta value: {optimal_beta}")
            return None
        
        theoretical_distortion = 1 / optimal_beta
        
        return theoretical_distortion   
    
    def find_violated_among_candidates(self, beta_hat: float, c_chosen):
            violated_cuts_data = []
        
            model = self.model
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', '')

            for k_idx, c_k in enumerate(self.candidates):
                objective_vector = c_chosen - beta_hat * c_k
                
                gurobi_obj_expr = gp.quicksum(objective_vector[dim] * self.vbar_vars[dim]
                                            for dim in range(self.d))
                model.setObjective(gurobi_obj_expr, gp.GRB.MINIMIZE)

                try:
                    model.optimize()

                    if model.status == gp.GRB.OPTIMAL:
                        min_obj_val = model.ObjVal
                        v_bar_solution_val = np.array([self.vbar_vars[dim].X for dim in range(self.d)])

                        if min_obj_val < -self.EPS_OBJ_VIOLATION:
                            violated_cuts_data.append((v_bar_solution_val, c_k))
                    
                    elif model.status == gp.GRB.INF_OR_UNBD:
                        print(f"Warning: Gurobi separation oracle for c_k (index {k_idx}) returned INF_OR_UNBD. Status: {model.status}.")
                    else:
                        print(f"Warning: Gurobi separation oracle for c_k (index {k_idx}) returned non-optimal status: {model.status}.")

                except gp.GurobiError as e:
                    print(f"Gurobi Error in find_violated_constraints for c_k (index {k_idx}): {e}. Skipping this candidate.")
                except Exception as e:
                    print(f"An unexpected error occurred during Gurobi optimization for c_k (index {k_idx}): {e}. Skipping this candidate.")
                    
            return violated_cuts_data    

    def calculate_theoretical_distortion(self, c_chosen, max_iter=100, tolerance=1e-6, verbose=False):
        master_theo_model = gp.Model("MasterLP")
        master_theo_model.setParam('OutputFlag', 0)
        master_theo_model.setParam('LogFile', '')
        beta_var = master_theo_model.addVar(name="beta", lb=0.0, ub=1.0)
        master_theo_model.setObjective(beta_var, gp.GRB.MAXIMIZE)
        
        optimal_beta = None  # Initialize optimal_beta
        
        for iteration in range(1, max_iter + 1):
            if verbose:
                print(f"\nIteration {iteration}: Solving Master Problem with {master_theo_model.NumConstrs} constraints...")
            
            try:
                master_theo_model.optimize()
                
                if master_theo_model.status not in [gp.GRB.OPTIMAL, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED]:
                    print(f"Master problem failed to solve or is not optimal at iteration {iteration}: {master_theo_model.status}")
                    return None
                
                # If master model is unbounded or infeasible, it's problematic
                if master_theo_model.status == gp.GRB.INF_OR_UNBD or master_theo_model.status == gp.GRB.UNBOUNDED:
                    print(f"Master problem is {master_theo_model.status} at iteration {iteration}. Returning None.")
                    return None
                
                current_beta_hat = beta_var.X
                if current_beta_hat is None:
                    print(f"Solver returned None values at iteration {iteration}. Status: {master_theo_model.status}")
                    return None
                
                if verbose:
                    print(f"Master solution: beta_hat={current_beta_hat:.6f}")
                
                # Call the separation oracle to find violated constraints
                # This will use the model (the other Gurobi model)
                violated_cuts_data = self.find_violated_among_candidates(current_beta_hat, c_chosen)
                
                if not violated_cuts_data:
                    if verbose:
                        print("No violated constraints found. Optimal solution reached.")
                    optimal_beta = current_beta_hat
                    break
                else:
                    for v_star, c_k_violated in violated_cuts_data:
                        # The cut is: <c_chosen, v_star> - beta_var * <c_k_violated, v_star> >= 0
                        # This is a linear constraint in beta_var
                        lhs_constant = np.dot(c_chosen, v_star)
                        rhs_coeff = np.dot(c_k_violated, v_star)
                        
                        # Add the new constraint to the master problem
                        # np.where is used to get the index of c_k_violated for naming the cut
                        # This assumes candidates are unique for reliable indexing.
                        c_k_idx = np.where((self.candidates == c_k_violated).all(axis=1))[0][0]
                        master_theo_model.addConstr(lhs_constant - beta_var * rhs_coeff >= 0,
                                                name=f"cut_iter{iteration}_ck{c_k_idx}_v{iteration}")
                    
                    master_theo_model.update()
                    
                    if verbose:
                        print(f"Found {len(violated_cuts_data)} new cut(s). Total cuts: {master_theo_model.NumConstrs}")
            
            except gp.GurobiError as e:
                print(f"Gurobi Error in Master Problem at iteration {iteration}: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred at iteration {iteration}: {e}")
                return None
        
        else:
            if verbose:
                print(f"Maximum iterations ({max_iter}) reached without full convergence.")
            # If we exit the loop without finding optimal solution, use the last beta value
            if 'current_beta_hat' in locals():
                optimal_beta = current_beta_hat
            else:
                print("No valid solution found.")
                return None
        
        # Check if optimal_beta is valid before computing theoretical distortion
        if optimal_beta is None or optimal_beta <= 0:
            print(f"Invalid optimal_beta value: {optimal_beta}")
            return None
        
        theoretical_distortion = 1 / optimal_beta
        
        return theoretical_distortion   
    
    def compute_linear_stable_lottery(self):
        output=np.zeros(self.d)
        k_committee_size = int(np.sqrt(self.d))
        found_lottery_committees, message_committees = find_stable_lottery(self.utilities, k_committee_size)

        
        print("\nFound Stable Lottery Distribution over Committees:")
        for committee, prob in found_lottery_committees.items():
            print(f"  Committee {committee}: Probability {prob:.4f}")

            #print(f"\n--- Verifying the found committee lottery ---")
            #is_stable_committees, verification_results_committees = check_stable_lottery(self.utilities, k_committee_size, found_lottery_committees)
            #print(f"Is the found committee lottery stable? {is_stable_committees}")
            #for alt, res in verification_results_committees.items():
            #    print(f"  {alt}: Expected |S_a(W)| = {res['expected_SaW']:.4f}, Threshold = {res['threshold']:.4f}, Condition Met: {res['condition_met']}")


        lslr_candidates, lslr_message = compute_lslr_candidate_distribution(found_lottery_committees, self.m, k_committee_size)
        
        
        for candidate_idx, prob in lslr_candidates.items():
            print(f"  Candidate {candidate_idx}: Probability {prob:.4f}")
            output+=prob*np.array(self.candidates[candidate_idx])
        print(f"  Sum of candidate probabilities: {sum(lslr_candidates.values())==0.5}")
        output+=0.5*np.ones(self.d)/self.d

        return output

    def find_violated_constraints(self, beta_hat: float, p_hat: np.ndarray) -> List[Tuple[float, gp.LinExpr]]:
            """
            Identifies violated constraints for the master problem given current beta_hat and p_hat.
            This acts as the separation oracle. It uses the pre-built Gurobi model
            and returns Gurobi linear expressions for the cuts.

            Args:
                beta_hat (float): Current value of beta from the master problem.
                p_hat (np.ndarray): Current probability distribution over candidates from the master problem.

            Returns:
                List[Tuple[float, gp.LinExpr]]: A list of tuples, where each tuple contains
                    (constant_term, Gurobi_linear_expression_for_p_vars).
                    The cut will be added to the master model as:
                    beta_var * constant_term - Gurobi_linear_expression_for_p_vars <= 0
            """
            violated_cuts_data = []
        
            if not p_hat.shape == (self.m,):
                raise ValueError(f"p_hat must be of shape ({self.m},). Got {p_hat.shape}")

            model = self.model
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', '')

            sum_pi_ci_vector = np.sum(p_hat[:, np.newaxis] * self.candidates, axis=0)

            for k_idx, c_k in enumerate(self.candidates):
                objective_vector = sum_pi_ci_vector - beta_hat * c_k
                
                gurobi_obj_expr = gp.quicksum(objective_vector[dim] * self.vbar_vars[dim]
                                            for dim in range(self.d))
                model.setObjective(gurobi_obj_expr, gp.GRB.MINIMIZE)

                try:
                    model.optimize()

                    if model.status == gp.GRB.OPTIMAL:
                        min_obj_val = model.ObjVal
                        v_bar_solution_val = np.array([self.vbar_vars[dim].X for dim in range(self.d)])

                        if min_obj_val < -self.EPS_OBJ_VIOLATION:
                            # Constructing the cut: beta_var * (c_k . v_bar_solution_val) - sum_j p_j * (C[j] . v_bar_solution_val) <= 0
                            
                            # The coefficient for beta_var in the cut
                            beta_coeff_in_cut = np.dot(c_k, v_bar_solution_val)
                            
                            # The linear expression for p_vars in the cut: sum_j p_j * (C[j] . v_bar_solution_val)
                            p_lin_expr_in_cut = gp.LinExpr()
                            for j in range(self.m):
                                p_lin_expr_in_cut.addTerms(np.dot(self.candidates[j], v_bar_solution_val), self.p_vars[j])
                            
                            violated_cuts_data.append((beta_coeff_in_cut, p_lin_expr_in_cut))
                    
                    elif model.status == gp.GRB.INF_OR_UNBD:
                        print(f"Warning: Gurobi separation oracle for c_k (index {k_idx}) returned INF_OR_UNBD. Status: {model.status}.")
                    else:
                        print(f"Warning: Gurobi separation oracle for c_k (index {k_idx}) returned non-optimal status: {model.status}.")

                except gp.GurobiError as e:
                    print(f"Gurobi Error in find_violated_constraints for c_k (index {k_idx}): {e}. Skipping this candidate.")
                except Exception as e:
                    print(f"An unexpected error occurred during Gurobi optimization for c_k (index {k_idx}): {e}. Skipping this candidate.")
                    
            return violated_cuts_data
        
    def calculate_optimal_random(self,max_iter=100, tolerance=1e-6, verbose=False):
        self.master_model = gp.Model("MasterLP")
        self.master_model.setParam('OutputFlag', 0)
        self.master_model.setParam('LogFile', '')

        # Define Gurobi variables for the master problem
        self.p_vars = self.master_model.addVars(self.m, name="p", lb=0.0, ub=1.0)
        self.beta_var = self.master_model.addVar(name="beta", lb=0.0, ub=1.0) # Assuming distortion >= 1, so beta <= 1

        # Add initial master constraints
        self.master_model.addConstr(gp.quicksum(self.p_vars[j] for j in range(self.m)) == 1, name="sum_p_equals_1")
        # p_vars >= 0 and beta_var >= 0, beta_var <= 1 are already handled by lb/ub

        # Set the objective: Maximize beta_var
        self.master_model.setObjective(self.beta_var, gp.GRB.MAXIMIZE)

        optimal_beta = None
        optimal_p = None
        
        if self.voters is None or self.candidates is None:
            raise ValueError("Voters, Candidates, or C matrix not initialized. Call generate_instance first.")

        if self.model is None: # Ensure separation oracle model is ready
            raise ValueError("Separation oracle Gurobi model not warm-started. Call warm_start_separation_oracle first.")

        for iteration in range(1, max_iter + 1):
            if verbose:
                print(f"\nIteration {iteration}: Solving Master Problem with {self.master_model.NumConstrs} constraints...")

            try:
                self.master_model.optimize()

                if self.master_model.status not in [gp.GRB.OPTIMAL, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED]:
                    print(f"Master problem failed to solve or is not optimal at iteration {iteration}: {self.master_model.status}")
                    return None, None, iteration
                
                # If master model is unbounded or infeasible, it's problematic
                if self.master_model.status == gp.GRB.INF_OR_UNBD or self.master_model.status == gp.GRB.UNBOUNDED:
                     print(f"Master problem is {self.master_model.status} at iteration {iteration}. Returning None.")
                     return None, None, iteration

                current_beta_hat = self.beta_var.X
                current_p_hat = np.array([self.p_vars[j].X for j in range(self.m)])

                if current_beta_hat is None or current_p_hat is None:
                    print(f"Solver returned None values at iteration {iteration}. Status: {self.master_model.status}")
                    return None, None, iteration

                if verbose:
                    print(f"Master solution: beta_hat={current_beta_hat:.6f}, p_hat={current_p_hat[:min(5, len(current_p_hat))]}...") 

                # Call the separation oracle to find violated constraints
                # This will use the model (the other Gurobi model)
                violated_cuts_data = self.find_violated_constraints(current_beta_hat, current_p_hat)

                if not violated_cuts_data:
                    if verbose:
                        print("No violated constraints found. Optimal solution reached.")
                    optimal_beta = current_beta_hat
                    optimal_p = current_p_hat
                    break
                else:
                    # Add new cuts to the Gurobi master model
                    for beta_coeff, p_lin_expr in violated_cuts_data:
                        # The cut is: beta_var * beta_coeff - p_lin_expr <= 0
                        self.master_model.addConstr(self.beta_var * beta_coeff - p_lin_expr <= 0, 
                                                    name=f"cut_iter{iteration}_{len(self.master_model.getConstrs())}")
                    if verbose:
                        print(f"Found {len(violated_cuts_data)} new cut(s). Total cuts: {self.master_model.NumConstrs}")     

            except gp.GurobiError as e:
                print(f"Gurobi Error in Master Problem at iteration {iteration}: {e}")
                return None, None, iteration
            except Exception as e:
                print(f"An unexpected error occurred at iteration {iteration}: {e}")
                return None, None, iteration
        else:
            if verbose:
                print(f"Maximum iterations ({max_iter}) reached without full convergence.")
            optimal_beta = current_beta_hat
            optimal_p = current_p_hat

        if optimal_beta is None or optimal_beta <= 0:
            theoretical_distortion = float('inf')
        else:
            theoretical_distortion = 1.0 / optimal_beta
        
        empirical_distortion = None
        if optimal_p is not None and self.utilities is not None:
            expected_social_welfare = np.dot(optimal_p, np.sum(self.utilities, axis=0))
            empirical_distortion = self.max_social_welfare / expected_social_welfare

        return theoretical_distortion, empirical_distortion, iteration

    def calculate_optimal_det(self, verbose: bool = False) -> Tuple[Optional[float], Optional[np.ndarray]]:
        # Initialize distortion matrix with infinity
        distortion_array=np.zeros(self.m)
        if verbose:
            print("\nCalculating worst-case distortion matrix (m x m)...")

        for i in range(self.m): # c_i is the numerator candidate (c*)
            distortion_array[i]=self.calculate_theoretical_distortion(self.candidates[i])

        
        if verbose:
            print("\nDistortion Array (max_ratio(c_i, c_j)):")
            print(np.array2string(distortion_array, precision=4, suppress_small=True))

        optimal_candidate_idx = np.argmin(distortion_array)
        theoretical_distortion = distortion_array[optimal_candidate_idx]
        empirical_distortion= self.max_social_welfare/ np.sum(self.utilities[:,optimal_candidate_idx])

        if verbose:
            print(f"\nOptimal Deterministic Candidate Index: {optimal_candidate_idx}")

        return  empirical_distortion, theoretical_distortion

    def calculate_distortion(self, rule: str):
        """
        Calculates empirical and theoretical distortion for a given social choice rule.

        Args:
            rule (str): The social choice rule to evaluate ('plurality', 'borda', 'mcp',
                        'random_dictatorship', 'random_harm', 'iod', 'ior').

        Returns:
            Tuple[float, float]: A tuple containing empirical distortion and theoretical distortion.
        """
        num_voters = self.n
        num_alternatives = self.m
        
        empirical_distortion = 0.0
        theoretical_distortion = 0.0
        winning_alternative_index = -1 # Index of the chosen candidate
        c_chosen_embedding = None # Embedding of the chosen candidate

        starting_t=time.time()
        if rule == 'plurality':
           
            votes = np.zeros(num_alternatives)
            for i in range(num_voters):
                most_preferred_alternative = np.argmax(self.utilities[i, :])
                votes[most_preferred_alternative] += 1
            
            winning_alternative_index = np.argmax(votes)
            finishing_t=time.time()-starting_t
            c_chosen_embedding = self.candidates[winning_alternative_index]
            
        elif rule == 'borda':
            borda_scores = np.zeros(num_alternatives)
            for i in range(num_voters):
                # Sort alternatives by utility in descending order to get ranks
                # argsort gives indices of sorted values. [::-1] reverses to descending order.
                ranked_alternatives_indices = np.argsort(self.utilities[i, :])[::-1]
                for rank, alt_index in enumerate(ranked_alternatives_indices):
                    # Borda score: num_alternatives - 1 - rank (0-indexed rank)
                    borda_scores[alt_index] += (num_alternatives - 1 - rank)
            
            winning_alternative_index = np.argmax(borda_scores)
            finishing_t=time.time()-starting_t
            c_chosen_embedding = self.candidates[winning_alternative_index]
            
        elif rule == "mcp": # Max Coordinate Product (or similar interpretation)
            if self.d > self.m:
                print(f"Warning: 'mcp' rule, d ({self.d}) is greater than m ({self.m}). "
                      "Cannot pick {self.d} distinct candidates based on d dimensions.")
                # Fallback or error handling
                # For now, let's just pick min(d, m) candidates based on the largest dimensions.
                num_dims_to_consider = min(self.d, self.m)
            else:
                num_dims_to_consider = self.d

            chosen_candidate_indices = []
            for dim in range(num_dims_to_consider):
                # Find the candidate that maximizes the value in this specific dimension
                max_cand_idx_for_dim = np.argmax(self.candidates[:, dim])
                chosen_candidate_indices.append(max_cand_idx_for_dim)
            
            # Remove duplicates if any candidate maximizes multiple dimensions
            chosen_candidate_indices = list(set(chosen_candidate_indices))

            # Now, from these chosen candidates, pick the "most popular" one.
            if not chosen_candidate_indices:
                print("Warning: No candidates selected for MCP rule. Defaulting to random choice.")
            else:
                conditional_votes = {idx: 0 for idx in chosen_candidate_indices} # Initialize votes for selected candidates

                for k in range(num_voters):
                    # Get utilities for only the selected candidates
                    utilities_for_selected_cands = {
                        idx: self.utilities[k, idx] for idx in chosen_candidate_indices
                    }
                    # Find the candidate with max utility among these selected candidates for voter k
                    if utilities_for_selected_cands: # Ensure there are candidates to choose from
                        best_candidate_for_voter = max(utilities_for_selected_cands, 
                                                       key=utilities_for_selected_cands.get)
                        conditional_votes[best_candidate_for_voter] += 1
                
            winning_alternative_index = max(conditional_votes, key=lambda idx: conditional_votes[idx])
            finishing_t=time.time()-starting_t
            c_chosen_embedding = self.candidates[winning_alternative_index]

        elif rule == 'random_dictatorship':
            # In random dictatorship, a random voter's top choice is chosen.
            # The "chosen candidate" for theoretical distortion is the average of the dictator's choices.
            # For empirical distortion, it's the expected social welfare.
            
            expected_social_welfare = 0.0
            average_chosen_embedding = np.zeros(self.d) # To average the chosen candidates' embeddings

            for i in range(num_voters):
                dictator_choice_idx = np.argmax(self.utilities[i, :]) # Get the i-th voter's top choice
                expected_social_welfare += np.sum(self.utilities[:, dictator_choice_idx]) # Sum utilities for that choice
                average_chosen_embedding += self.candidates[dictator_choice_idx] # Accumulate embeddings

            # Average social welfare for empirical distortion
            empirical_distortion_val = self.max_social_welfare / (expected_social_welfare / num_voters)
            
            # Average chosen embedding for theoretical distortion
            c_chosen_embedding = average_chosen_embedding / num_voters
            finishing_t=time.time()-starting_t
            
            empirical_distortion = empirical_distortion_val # Assign to the main variable

        elif rule == 'random_harm':
            # This rule usually involves probabilistic choice based on harmonic scores,
            # then calculating expected social welfare and the expected chosen candidate embedding.
            harmonic_scores = np.zeros(num_alternatives)
            for k in range(num_voters): # For each voter
                # Get sorted indices of candidates by utility for voter k (descending)
                ranked_indices = np.argsort(-self.utilities[k, :])
                for rank, cand_idx in enumerate(ranked_indices):
                    harmonic_scores[cand_idx] += 1 / (rank + 1) # Rank is 0-indexed, so add 1

            # Normalize scores to get probabilities
            total_score = np.sum(harmonic_scores)
            if total_score == 0:
                probabilities = np.ones(num_alternatives) / num_alternatives # Uniform random if all scores are zero
            else:
                probabilities = harmonic_scores / total_score

            # Calculate the expected chosen candidate embedding
            c_chosen_embedding = np.sum(probabilities[:, np.newaxis] * self.candidates, axis=0)
            finishing_t=time.time()-starting_t
            # Calculate expected social welfare for empirical distortion
            expected_social_welfare_chosen = 0.0
            for alt_idx in range(num_alternatives):
                expected_social_welfare_chosen += probabilities[alt_idx] * np.sum(self.utilities[:, alt_idx])
            
            if expected_social_welfare_chosen == 0:
                empirical_distortion = float('inf') # Avoid division by zero
            else:
                empirical_distortion = self.max_social_welfare / expected_social_welfare_chosen

        # Calculate theoretical distortion if c_chosen_embedding is available
        starting_t_2=time.time()
        if c_chosen_embedding is not None:
            theoretical_distortion = self.calculate_theoretical_distortion(c_chosen_embedding)
        else:
            print("candidate was not found")
            theoretical_distortion = float('inf') # If no candidate was chosen for some reason
        checking_t=time.time()-starting_t_2

        # If rule was random_dictatorship or random_harm, empirical_distortion was already calculated
        # If it was plurality, borda, or mcp, calculate it now based on winning_alternative_index
        if rule in ['plurality', 'borda', 'mcp']:
            if winning_alternative_index != -1:
                social_welfare_chosen = np.sum(self.utilities[:, winning_alternative_index])
                if social_welfare_chosen == 0:
                    empirical_distortion = float('inf')
                else:
                    empirical_distortion = self.max_social_welfare / social_welfare_chosen
            else:
                empirical_distortion = float('inf') # Should not happen if a winner is always selected

        return empirical_distortion, theoretical_distortion, finishing_t, checking_t

    def solve_max_min_c_v_problem(self, max_iter: int = 100, tolerance: float = 1e-6, verbose: bool = False) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Solves the problem: max_c (min_v <c, v>) subject to v in a polytope (feasible region for vbar)
        and c in a simplex. This is implemented using a cutting-plane approach.

        Args:
            max_iter (int): Maximum number of iterations for the cutting plane algorithm.
            tolerance (float): Tolerance for convergence.
            verbose (bool): If True, print detailed progress.

        Returns:
            Tuple[Optional[float], Optional[np.ndarray]]:
                - The optimal value of the objective (max_c min_v <c,v>).
                - The optimal c vector (embedding).
        """
        if self.model is None:
            raise ValueError("Separation oracle Gurobi model not warm-started. Call warm_start_separation_oracle first.")

        # 1. Initialize the Master Problem
        master_model_max_min = gp.Model("MaxMin_c_v")
        master_model_max_min.setParam('OutputFlag', 0)
        master_model_max_min.setParam('LogFile', '')

        # Variables for the Master Problem
        c_vars_max_min = master_model_max_min.addVars(self.d, name="c", lb=0.0, ub=1.0) # c in R^d
        gamma_var_max_min = master_model_max_min.addVar(name="gamma", lb=0.0, ub=1.0) # Objective variable

        # Constraints for c (simplex)
        master_model_max_min.addConstr(gp.quicksum(c_vars_max_min[i] for i in range(self.d)) == 1, name="c_simplex_l1_norm")

        # Objective: Maximize gamma
        master_model_max_min.setObjective(gamma_var_max_min, gp.GRB.MAXIMIZE)

        optimal_gamma = None
        optimal_c = None

        # Iteration loop for cutting plane
        for iteration in range(1, max_iter + 1):
            if verbose:
                print(f"\nMax-Min Iteration {iteration}: Solving Master Problem with {master_model_max_min.NumConstrs} constraints...")

            try:
                master_model_max_min.optimize()

                if master_model_max_min.status not in [gp.GRB.OPTIMAL, gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED]:
                    print(f"Master problem failed to solve or is not optimal at iteration {iteration}: {master_model_max_min.status}")
                    return None, None
                
                if master_model_max_min.status == gp.GRB.INF_OR_UNBD or master_model_max_min.status == gp.GRB.UNBOUNDED:
                    print(f"Master problem is {master_model_max_min.status} at iteration {iteration}. Returning None.")
                    return None, None

                current_c_hat = np.array([c_vars_max_min[i].X for i in range(self.d)])
                current_gamma_hat = gamma_var_max_min.X

                if verbose:
                    print(f"  Current c_hat: {current_c_hat[:min(5,self.d)]}..., gamma_hat: {current_gamma_hat:.6f}")
            
                separation_obj_expr = gp.quicksum(current_c_hat[dim] * self.vbar_vars[dim] for dim in range(self.d))
                self.model.setObjective(separation_obj_expr, gp.GRB.MINIMIZE)
          
                self.model.optimize()
                if self.model.status == gp.GRB.OPTIMAL:
                    min_val_from_oracle = self.model.ObjVal
                    vbar_solution_for_cut = np.array([self.vbar_vars[dim].X for dim in range(self.d)])
                else:
                    print(f"Warning: Separation oracle in max-min failed (status: {self.model.status}).")
                    return None, None

                if verbose:
                    print(f"  Min <c_hat, vbar> from oracle: {min_val_from_oracle:.6f}")

                # 3. Check for violation and add cut
                if current_gamma_hat > min_val_from_oracle + tolerance: # Violation found
                    if verbose:
                        print(f"    Violation found. Adding new cut: gamma <= <c, vbar*>")
                    # Add the cut: gamma <= c^T * vbar_solution_for_cut
                    cut_expr = gp.quicksum(c_vars_max_min[dim] * vbar_solution_for_cut[dim] for dim in range(self.d))
                    master_model_max_min.addConstr(gamma_var_max_min <= cut_expr, name=f"cut_max_min_iter{iteration}_{master_model_max_min.NumConstrs}")
                else: # No significant violation, converged
                    if verbose:
                        print("    No significant violation found. Convergence achieved.")
                    optimal_gamma = current_gamma_hat
                    optimal_c = current_c_hat
                    break

            except gp.GurobiError as e:
                print(f"Gurobi Error in Max-Min Master Problem at iteration {iteration}: {e}")
                return None, None
            except Exception as e:
                print(f"An unexpected error occurred in Max-Min algorithm at iteration {iteration}: {e}")
                return None, None
        else:
            if verbose:
                print(f"Max-Min algorithm reached max iterations ({max_iter}) without full convergence.")
            optimal_gamma = current_gamma_hat
            optimal_c = current_c_hat

        return optimal_gamma, optimal_c

    def distortion_comparisons(self):
        print("\n--- Distortion Comparisons ---")
        rules_to_compare = ["plurality", "borda", "mcp", "random_dictatorship", "random_harm"] # Removed 'iod', 'ior' as they are not implemented

        for rule in rules_to_compare:
            print(f"  Rule: {rule.replace('_', ' ').title()}")
            emp_dist, theo_dist, r_time,theo_time = self.calculate_distortion(rule)
            if emp_dist is not None and theo_dist is not None:
                print(f"    Empirical Distortion: {emp_dist:.4f}")
                print(f"    Theoretical Distortion: {theo_dist:.4f}")
                print(f"    Running Time: {r_time:.4f}")
                print(f"    Theoretical Bound Calculating Time: {theo_time:.4f}")
            else:
                print(f"    Could not calculate distortion for {rule}.")

        print(f"  Rule: Randomized Instance Optimal")
        start_t=time.time()
        theo_dist, emp_dist, iterations = self.calculate_optimal_random()
        finish_t=time.time()-start_t
        if emp_dist is not None and theo_dist is not None:
            print("    Iterations",iterations)
            print(f"    Empirical Distortion: {emp_dist:.4f}")
            print(f"    Theoretical Distortion: {theo_dist:.4f}")
            print(f"    Running Time: {finish_t:.4f}")
        else:
            print(f"    Could not calculate distortion for {rule}.")
        
        print(f"  Rule: Deterministic Instance Optimal")
        start_t=time.time()
        emp_dist, theo_dist = self.calculate_optimal_det()
        finish_t=time.time()-start_t
        if emp_dist is not None and theo_dist is not None:
            print(f"    Empirical Distortion: {emp_dist:.4f}")
            print(f"    Theoretical Distortion: {theo_dist:.4f}")
            print(f"    Running Time: {finish_t:.4f}")
        else:
            print(f"    Could not calculate distortion for {rule}.")

        print(f"  Rule: uniform")
        uniform_c=np.ones(self.d)/self.d
        emp_dist = self.max_social_welfare * self.d / self.n
        theo_dist=self.calculate_theoretical_distortion(np.array(uniform_c))
        print(f"    Empirical Distortion: {emp_dist:.4f}")
        print(f"    Theoretical Distortion: {theo_dist:.4f}")


        print(f" Rule: Linear Stable Lottery Rule (LSLR)")
        start_t=time.time()
        lslr_c = self.compute_linear_stable_lottery()
        finish_t=time.time()-start_t
        if lslr_c is not None:
            emp_dist = self.max_social_welfare / (self.voters @ lslr_c).sum()
            theo_dist = self.calculate_theoretical_distortion(np.array(lslr_c))
            print(f"    Running Time: {finish_t:.4f}")
            print(f"    Empirical Distortion: {emp_dist:.4f}")
            print(f"    Theoretical Distortion: {theo_dist:.4f}")
        else:
            print(f"    Could not compute Linear Stable Lottery Rule (LSLR) candidate distribution.")
   
