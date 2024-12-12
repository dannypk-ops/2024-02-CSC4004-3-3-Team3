import math
from itertools import combinations

# Function to calculate the distance matrix
def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance_matrix[i][j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance_matrix

# Held-Karp TSP algorithm
def tsp_held_karp(coordinates):
    n = len(coordinates)
    distance_matrix = calculate_distance_matrix(coordinates)

    start_point = 0  # Explicitly set the starting point to the first city

    # Initialize DP table
    dp = {}
    for i in range(1, n):
        dp[(1 << i, i)] = distance_matrix[start_point][i]  # Cost from start_point to i

    # Iterate over subsets of increasing size
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            subset_mask = sum(1 << i for i in subset)
            for j in subset:
                prev_mask = subset_mask & ~(1 << j)
                dp[(subset_mask, j)] = float('inf')  # Initialize
                dp[(subset_mask, j)] = min(
                    dp[(prev_mask, k)] + distance_matrix[k][j]
                    for k in subset if k != j
                )

    # Calculate minimum cost to complete the cycle
    all_visited_mask = (1 << n) - 1
    min_cost = min(
        dp[(all_visited_mask & ~(1 << start_point), i)] + distance_matrix[i][start_point]
        for i in range(1, n)
    )

    # Reconstruct the optimal route
    optimal_route = [start_point]
    current_mask = all_visited_mask & ~(1 << start_point)
    last_vertex = min(
        range(1, n),
        key=lambda i: dp.get((current_mask, i), float('inf')) + distance_matrix[i][start_point],
    )
    optimal_route.append(last_vertex)

    for _ in range(n - 2):
        next_mask = current_mask & ~(1 << last_vertex)
        if next_mask == 0:  # Break if no more cities to visit
            break
        next_vertex = min(
            (i for i in range(1, n) if (current_mask & (1 << i))),
            key=lambda i: dp.get((next_mask, i), float('inf')) + distance_matrix[i][last_vertex],
        )
        optimal_route.append(next_vertex)
        current_mask = next_mask
        last_vertex = next_vertex

    optimal_route.append(start_point)  # Complete the cycle back to the start
    return optimal_route, min_cost

if __name__ == "__main__":
    # Define the coordinates as a list of tuples
    coordinates = [
        (-2.0, -0.5),
        (-2.05, -0.5),
        (0.95, -1.6),
        (-1.35, -1.85),
        (1.25, -1.65),
        (0.9, 0.45),
        (0.55, -1.65),
        (-2.15, 0.6),
        (-1.75, -0.25),
        (2.15, -0.15),
        (0.65, -0.95),
        (0.9, -1.9),
        (1.4, -0.5),
        (0.75, 0.95),
        (2.5, -0.8),
        (-0.9, -0.5),
        (-0.4, -0.1),
        (-0.2, 0.2),
        (1.95, 0.9),
        (0.25, 0.45),
    ]

    optimal_route, min_cost = tsp_held_karp(coordinates)

    # Print the optimal route and its cost
    print(f"Optimal Route: {optimal_route}")
    print(f"Total Cost: {min_cost:.2f}")
