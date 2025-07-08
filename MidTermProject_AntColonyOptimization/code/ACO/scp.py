import numpy as np
import random
import matplotlib.pyplot as plt

# Problem Parameters
n_rows = 10  # Number of rows (sets to be covered)
n_cols = 5  # Number of columns (sets available)
cost_matrix = np.random.randint(1, 20, n_cols)  # Random costs for each column
coverage_matrix = np.random.randint(0, 2, (n_rows, n_cols))  # Binary matrix of coverage

# ACO Parameters
alpha = 1.0  # Influence of pheromone
beta = 2.0  # Influence of heuristic (cost)
rho = 0.1  # Pheromone evaporation rate
Q = 100  # Total pheromone to deposit
n_ants = 30  # Number of ants
n_iterations = 100  # Number of iterations
epsilon = 1e-6  # Small constant to prevent division by zero

# Initialize pheromone matrix
pheromone_matrix = np.ones((n_rows, n_cols))  # Pheromone for row-column assignments


# ACO Algorithm
def aco_scp():
    global pheromone_matrix  # Access the global pheromone matrix

    best_solution = None
    best_cost = float('inf')

    for iteration in range(n_iterations):
        solutions = []  # List to store solutions (row assignments) for all ants
        costs = []  # List to store costs for all ants

        # Each ant constructs a solution (set of columns selected)
        for ant in range(n_ants):
            solution = construct_solution()
            cost = calculate_cost(solution)
            solutions.append(solution)
            costs.append(cost)

            # Update the best solution if necessary
            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        # Pheromone update
        pheromone_matrix *= (1 - rho)  # Evaporation
        for i in range(n_ants):
            for j in range(n_cols):
                if solutions[i][j] == 1:  # Update pheromone if column j is selected
                    pheromone_matrix[:, j] += Q / costs[i]

        # Print progress
        print(f"Iteration {iteration + 1}/{n_iterations} - Best Cost: {best_cost}")

    return best_solution, best_cost


# Function to construct a solution for an ant (set of column assignments)
def construct_solution():
    solution = [0] * n_cols  # Initialize solution with no columns selected
    covered_rows = [False] * n_rows  # Track which rows are covered

    for _ in range(n_cols):
        next_col = select_next_column(covered_rows)
        solution[next_col] = 1
        # Mark the rows covered by this column
        for i in range(n_rows):
            if coverage_matrix[i, next_col] == 1:
                covered_rows[i] = True

    return solution


# Function to select the next column to add based on pheromone and cost
def select_next_column(covered_rows):
    probabilities = []
    for col in range(n_cols):
        if not all(covered_rows):
            pheromone = pheromone_matrix[:, col].sum() ** alpha
            heuristic = 1 / (cost_matrix[col] + epsilon)  # Avoid zero heuristic
            probabilities.append(pheromone * heuristic)
        else:
            probabilities.append(0)

    # Normalize the probabilities and handle NaN or zero-sum
    total_prob = sum(probabilities)
    if total_prob == 0:
        return random.choice(range(n_cols))  # If all probabilities are zero, pick randomly

    probabilities = [p / total_prob for p in probabilities]

    # Handle NaN probabilities by replacing them with zeros
    probabilities = np.nan_to_num(probabilities, nan=0.0)

    # Select next column based on probabilities
    return np.random.choice(range(n_cols), p=probabilities)


# Function to calculate the total cost of a given solution (set of column selections)
def calculate_cost(solution):
    total_cost = 0
    covered_rows = [False] * n_rows

    for col in range(n_cols):
        if solution[col] == 1:
            total_cost += cost_matrix[col]  # Add the cost of this column
            for i in range(n_rows):
                if coverage_matrix[i, col] == 1:
                    covered_rows[i] = True

    # Add a large penalty if not all rows are covered
    if not all(covered_rows):
        total_cost += 1e6  # Add a large penalty for uncovered rows

    return total_cost


# Visualize the result: Display the coverage matrix
def visualize_solution(solution):
    coverage_matrix_display = np.zeros((n_rows, n_cols))

    for col in range(n_cols):
        if solution[col] == 1:
            coverage_matrix_display[:, col] = coverage_matrix[:, col]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.matshow(coverage_matrix_display, cmap='Blues')

    # Annotate the matrix with task numbers and column numbers
    for i in range(n_rows):
        for j in range(n_cols):
            if coverage_matrix_display[i, j] == 1:
                ax.text(j, i, f'C{j + 1}', ha='center', va='center', color='red')

    plt.xlabel("Columns (Sets)")
    plt.ylabel("Rows (Elements to cover)")
    plt.title("ACO Solution for Set Covering Problem (SCP)")
    plt.xticks(range(n_cols), [f"C{j + 1}" for j in range(n_cols)])
    plt.yticks(range(n_rows), [f"R{i + 1}" for i in range(n_rows)])
    plt.show()


# Run the ACO algorithm
best_solution, best_cost = aco_scp()

# Visualize the result
print(f"Best solution cost: {best_cost}")
visualize_solution(best_solution)
