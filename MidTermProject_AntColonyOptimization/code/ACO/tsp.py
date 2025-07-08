import numpy as np
import random
import matplotlib.pyplot as plt

# Problem Parameters (number of cities, coordinates of cities)
n_cities = 20  # Number of cities
cities = np.random.rand(n_cities, 2)  # Generate random 2D coordinates for cities

# ACO Parameters
alpha = 1.0  # Influence of pheromone
beta = 2.0  # Influence of distance (heuristic)
rho = 0.1  # Pheromone evaporation rate
Q = 100  # Total pheromone to deposit
n_ants = 50  # Number of ants
n_iterations = 100  # Number of iterations

# Distance matrix (Euclidean distance between cities)
dist_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])

# Initialize pheromone matrix globally here
pheromone_matrix = np.ones((n_cities, n_cities))


# ACO Algorithm
def aco_tsp():
    global pheromone_matrix  # Access the global pheromone matrix
    best_path = None
    best_length = float('inf')

    for iteration in range(n_iterations):
        paths = []  # List to store paths for all ants
        lengths = []  # List to store lengths for all ants

        # Each ant constructs a path
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
            for j in range(n_cities - 1):
                pheromone_matrix[paths[i][j], paths[i][j + 1]] += Q / lengths[i]
                pheromone_matrix[paths[i][j + 1], paths[i][j]] += Q / lengths[i]

        # Print progress
        print(f"Iteration {iteration + 1}/{n_iterations} - Best Length: {best_length}")

    return best_path, best_length


# Function to construct a path for an ant
def construct_path():
    path = [random.randint(0, n_cities - 1)]  # Start from a random city
    visited = [False] * n_cities
    visited[path[0]] = True

    for _ in range(n_cities - 1):
        current_city = path[-1]
        next_city = select_next_city(current_city, visited)
        path.append(next_city)
        visited[next_city] = True

    return path


# Function to select the next city based on pheromone and distance
def select_next_city(current_city, visited):
    probabilities = []
    for next_city in range(n_cities):
        if not visited[next_city]:
            pheromone = pheromone_matrix[current_city][next_city] ** alpha
            distance = dist_matrix[current_city][next_city] ** (-beta)
            probabilities.append(pheromone * distance)
        else:
            probabilities.append(0)

    # Normalize the probabilities
    total_prob = sum(probabilities)
    probabilities = [p / total_prob for p in probabilities]

    # Select next city based on probabilities
    return np.random.choice(range(n_cities), p=probabilities)


# Function to calculate the length of a given path
def calculate_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += dist_matrix[path[i], path[i + 1]]
    length += dist_matrix[path[-1], path[0]]  # Return to the starting city
    return length


# Visualize the TSP solution
def visualize(path):
    plt.figure(figsize=(8, 6))
    for i in range(len(path) - 1):
        plt.plot([cities[path[i]][0], cities[path[i + 1]][0]], [cities[path[i]][1], cities[path[i + 1]][1]], 'b-', lw=2)
    plt.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], 'b-',
             lw=2)  # Closing the loop

    plt.scatter(cities[:, 0], cities[:, 1], c='r', marker='o', s=100)
    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.01, y + 0.01, str(i), fontsize=12)

    plt.title("ACO TSP Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Run the ACO algorithm
best_path, best_length = aco_tsp()

# Visualize the result
print(f"Best path length: {best_length}")
visualize(best_path)
