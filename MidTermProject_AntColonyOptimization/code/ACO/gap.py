import numpy as np
import random
import matplotlib.pyplot as plt

# Problem Parameters
n_tasks = 10  # Number of tasks
n_agents = 5  # Number of agents

# Randomly generated cost matrix d_ij (cost of assigning task i to agent j)
cost_matrix = np.random.randint(1, 20, (n_tasks, n_agents))

# Randomly generated capacity usage matrix b_ij (capacity used by task i on agent j)
capacity_matrix = np.random.randint(1, 10, (n_tasks, n_agents))

# Random agent capacities
agent_capacities = np.random.randint(20, 30, n_agents)

# ACO Parameters
alpha = 1.0  # Influence of pheromone
beta = 2.0  # Influence of heuristic (cost)
rho = 0.1  # Pheromone evaporation rate
Q = 100  # Total pheromone to deposit
n_ants = 30  # Number of ants
n_iterations = 100  # Number of iterations
epsilon = 1e-6  # Small constant to prevent division by zero

# Initialize pheromone matrix
pheromone_matrix = np.ones((n_tasks, n_agents))  # Pheromone for task-agent assignments


# ACO Algorithm
def aco_gap():
    global pheromone_matrix  # Access the global pheromone matrix

    best_path = None
    best_cost = float('inf')

    for iteration in range(n_iterations):
        paths = []  # List to store paths (task assignments) for all ants
        costs = []  # List to store costs for all ants

        # Each ant constructs a path (sequence of task assignments)
        for ant in range(n_ants):
            path = construct_path()
            cost = calculate_cost(path)
            paths.append(path)
            costs.append(cost)

            # Update the best path if necessary
            if cost < best_cost:
                best_path = path
                best_cost = cost

        # Pheromone update
        pheromone_matrix *= (1 - rho)  # Evaporation
        for i in range(n_ants):
            for j in range(n_tasks):
                # Ensure we don't exceed the available number of agents
                agent_index = paths[i][j]
                if agent_index < n_agents and j < n_agents:
                    pheromone_matrix[agent_index, j] += Q / costs[i]

        # Print progress
        print(f"Iteration {iteration + 1}/{n_iterations} - Best Cost: {best_cost}")

    return best_path, best_cost


# Function to construct a path for an ant (assignment of tasks to agents)
def construct_path():
    path = [-1] * n_tasks  # Initialize path with no assignments
    visited_agents = [False] * n_agents  # Track visited agents

    for task in range(n_tasks):
        next_agent = select_next_agent(task, visited_agents)
        path[task] = next_agent
        visited_agents[next_agent] = True  # Mark agent as visited

    return path


# Function to select the next agent for a given task based on pheromone and cost
def select_next_agent(task, visited_agents):
    probabilities = []
    for agent in range(n_agents):
        if not visited_agents[agent]:
            pheromone = pheromone_matrix[task][agent] ** alpha
            heuristic = 1 / (cost_matrix[task][agent] + epsilon)  # Avoid zero heuristic
            probabilities.append(pheromone * heuristic)
        else:
            probabilities.append(0)

    # Normalize the probabilities and handle NaN or zero-sum
    total_prob = sum(probabilities)
    if total_prob == 0:
        return random.choice(range(n_agents))  # If all probabilities are zero, pick randomly

    probabilities = [p / total_prob for p in probabilities]

    # Handle NaN probabilities by replacing them with zeros
    probabilities = np.nan_to_num(probabilities, nan=0.0)

    # Select next agent based on probabilities
    return np.random.choice(range(n_agents), p=probabilities)


# Function to calculate the total cost of a given assignment path
def calculate_cost(path):
    total_cost = 0
    total_capacity_used = np.zeros(n_agents)

    for task in range(n_tasks):
        agent = path[task]
        total_cost += cost_matrix[task][agent]
        total_capacity_used[agent] += capacity_matrix[task][agent]

    # If any agent exceeds its capacity, add a penalty to the cost
    for agent in range(n_agents):
        if total_capacity_used[agent] > agent_capacities[agent]:
            total_cost += 1e6  # Add a large penalty for exceeding capacity

    return total_cost


# Visualize the result: Display the assignment matrix
def visualize_assignment(path):
    assignment_matrix = np.zeros((n_tasks, n_agents))

    for task in range(n_tasks):
        agent = path[task]
        assignment_matrix[task, agent] = 1  # Mark the assignment

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.matshow(assignment_matrix, cmap='Blues')

    # Annotate the matrix with task numbers and agent numbers
    for i in range(n_tasks):
        for j in range(n_agents):
            if assignment_matrix[i, j] == 1:
                ax.text(j, i, f'T{i + 1}', ha='center', va='center', color='red')

    plt.xlabel("Agents")
    plt.ylabel("Tasks")
    plt.title("ACO Solution for Generalized Assignment Problem (GAP)")
    plt.xticks(range(n_agents), [f"A{j + 1}" for j in range(n_agents)])
    plt.yticks(range(n_tasks), [f"T{j + 1}" for j in range(n_tasks)])
    plt.show()


# Run the ACO algorithm
best_path, best_cost = aco_gap()

# Visualize the result
print(f"Best path cost: {best_cost}")
visualize_assignment(best_path)
