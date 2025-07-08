import numpy as np
import random
import matplotlib.pyplot as plt

# Problem Parameters (number of jobs, processing time, due date, weight)
n_jobs = 10  # Number of jobs
processing_times = np.random.randint(1, 20, n_jobs)  # Random processing times for jobs
due_dates = np.random.randint(10, 30, n_jobs)  # Random due dates for jobs
weights = np.random.randint(1, 10, n_jobs)  # Random weights for jobs

# ACO Parameters
alpha = 1.0  # Influence of pheromone
beta = 2.0  # Influence of priority (heuristic info)
rho = 0.1  # Pheromone evaporation rate
Q = 100  # Total pheromone to deposit
n_ants = 30  # Number of ants
n_iterations = 100  # Number of iterations
epsilon = 1e-6  # Small constant to prevent division by zero


# Heuristic information (modified due date heuristic)
def get_mdd(j):
    return max(processing_times[:j].sum() + processing_times[j], due_dates[j])


# ACO Initialization (Pheromone matrix)
pheromone_matrix = np.ones((n_jobs, n_jobs))  # Pheromone matrix for job ordering


# ACO Algorithm
def aco_smtwtp():
    global pheromone_matrix  # Access the global pheromone matrix
    best_path = None
    best_length = float('inf')

    for iteration in range(n_iterations):
        paths = []  # List to store paths for all ants
        lengths = []  # List to store lengths for all ants

        # Each ant constructs a path (sequence of jobs)
        for ant in range(n_ants):
            path = construct_path()
            length = calculate_path_length(path)
            paths.append(path)
            lengths.append(length)

            # Update the best path if necessary
            if length < best_length:
                best_path = path
                best_length = length

        # Pheromone update
        pheromone_matrix *= (1 - rho)  # Evaporation
        for i in range(n_ants):
            for j in range(n_jobs - 1):
                pheromone_matrix[paths[i][j], paths[i][j + 1]] += Q / lengths[i]

        # Print progress
        print(f"Iteration {iteration + 1}/{n_iterations} - Best Length: {best_length}")

    return best_path, best_length


# Function to construct a path for an ant (sequence of jobs)
def construct_path():
    path = [random.randint(0, n_jobs - 1)]  # Start from a random job
    visited = [False] * n_jobs
    visited[path[0]] = True

    for _ in range(n_jobs - 1):
        current_job = path[-1]
        next_job = select_next_job(current_job, visited)
        path.append(next_job)
        visited[next_job] = True

    return path


# Function to select the next job based on pheromone and distance (heuristic information)
def select_next_job(current_job, visited):
    probabilities = []
    for next_job in range(n_jobs):
        if not visited[next_job]:
            pheromone = pheromone_matrix[current_job][next_job] ** alpha

            # Calculate heuristic with protection for division by zero
            heuristic = get_mdd(next_job) - due_dates[next_job]
            if heuristic == 0:
                heuristic = epsilon  # Avoid division by zero
            heuristic = heuristic ** (-beta)

            probabilities.append(pheromone * heuristic)
        else:
            probabilities.append(0)

    # Normalize the probabilities and handle NaN
    total_prob = sum(probabilities)
    if total_prob == 0:
        return random.choice(range(n_jobs))  # If all probabilities are zero, pick a random job

    probabilities = [p / total_prob for p in probabilities]

    # Handle NaN probabilities, replacing them with zeros
    probabilities = np.nan_to_num(probabilities, nan=0.0)

    # Ensure probabilities sum to 1
    total_prob = sum(probabilities)
    if total_prob != 1:
        probabilities = [p / total_prob for p in probabilities]  # Normalize again

    # Select next job based on probabilities
    return np.random.choice(range(n_jobs), p=probabilities)


# Function to calculate the length of a given path (total weighted tardiness)
def calculate_path_length(path):
    completion_times = np.zeros(n_jobs)
    tardiness = np.zeros(n_jobs)

    completion_times[path[0]] = processing_times[path[0]]
    tardiness[path[0]] = max(0, completion_times[path[0]] - due_dates[path[0]])

    for i in range(1, len(path)):
        job = path[i]
        completion_times[job] = completion_times[path[i - 1]] + processing_times[job]
        tardiness[job] = max(0, completion_times[job] - due_dates[job])

    total_tardiness = np.sum(weights * tardiness)
    return total_tardiness


# Visualize the TSP solution (jobs on a time axis)
def visualize(path):
    start_time = 0
    job_completion_times = []

    plt.figure(figsize=(10, 6))
    for job in path:
        job_completion_times.append((start_time, start_time + processing_times[job]))
        start_time += processing_times[job]

    for i, (start, end) in enumerate(job_completion_times):
        plt.plot([i, i], [start, end], 'b-', lw=6)  # Job completion line
        plt.text(i, (start + end) / 2 + 10, f"Job {path[i]}", fontsize=12, ha='center')

    plt.title("ACO Solution for SMTWTP")
    plt.xlabel("Job Index")
    plt.ylabel("Time")
    plt.show()


# Run the ACO algorithm
best_path, best_length = aco_smtwtp()

# Visualize the result
print(f"Best path length: {best_length}")
visualize(best_path)
