import math
import samplingPoint

def is_point_inside_sector(coordinate, min_radius, max_radius, point, direction_vector, angle):
    x0, y0 = coordinate
    px, py = point

    # Calculate the squared distance from the coordinate to the point
    distance_squared = (px - x0) ** 2 + (py - y0) ** 2
    if distance_squared < min_radius ** 2 or distance_squared > max_radius ** 2:
        return False  # Outside the specified radius range

    # Vector from coordinate to the point
    vector_to_point = (px - x0, py - y0)
    magnitude = math.sqrt(vector_to_point[0] ** 2 + vector_to_point[1] ** 2)
    if magnitude == 0:  # Point is exactly at the coordinate
        return True

    # Normalize vector to point
    vector_to_point = (vector_to_point[0] / magnitude, vector_to_point[1] / magnitude)

    # Normalize direction_vector
    magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    direction_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)

    # Calculate the angle between the two vectors using the dot product
    dot_product = (direction_vector[0] * vector_to_point[0] +
                   direction_vector[1] * vector_to_point[1])
    dot_product = max(min(dot_product, 1), -1)  # Clamp to [-1, 1] to avoid rounding errors
    theta = math.degrees(math.acos(dot_product))

    # Check if the angle is within half the sector angle
    return theta <= angle / 2


def count_points_in_sector(file_path, coordinate, min_radius, max_radius, direction_vector, angle):
    count = 0

    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            if is_point_inside_sector(coordinate, min_radius, max_radius, (x, y), direction_vector, angle):
                count += 1

    return count


def check_eight_direction(file_path, coordinate, min_radius, max_radius, angle):
    # Define 8 direction vectors
    direction_vectors = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    max_points = 0
    best_direction = None

    for direction in direction_vectors:
        # Count points in the sector for each direction
        points_in_sector = count_points_in_sector(file_path, coordinate, min_radius, max_radius, direction, angle)
        if points_in_sector > max_points:
            max_points = points_in_sector
            best_direction = direction

    return best_direction, max_points


def find_top_N_coordinates(file_path, n, coordinates, min_radius, max_radius, angle):
    results = []

    for coordinate in coordinates:
        # Check 8 directions for each coordinate and find the one with the most points
        direction, point_count = check_eight_direction(file_path, coordinate, min_radius, max_radius, angle)
        results.append((coordinate, direction, point_count))

    # Sort results by the number of points in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # Extract the top N coordinates
    top_N = results[:n]
    return top_N


if __name__ == "__main__":

    # File path for coordinates to be analyzed
    coordinate_file_path = "/home/air/open_sw/coordinates.txt" 
    n = 10  # Number of top coordinates to retrieve
    min_radius = 0.5  # Minimum radius of the sector
    max_radius = 1.0  # Maximum radius of the sector
    angle = 45  # Angle of the sector in degrees

    # Parameters for sampling points
    map_file_path = "/home/air/open_sw/occupancyGridMap.txt"
    start_point = (-2.0, -0.5)  # Starting point
    divisions = (1, 1)  # Divide the grid into 4 regions for rows and columns respectively
    samples_per_region = 50  # Select 2 samples from each region
    origin = (52, 52)  # Origin index in the grid
    resolution = 0.05  # Actual distance between indices

    # Generate sample coordinates
    coordinates = samplingPoint.divide_and_sample_with_coordinates(map_file_path, divisions, samples_per_region, origin, resolution)

    # Find the top N coordinates
    top_N_coordinates = find_top_N_coordinates(coordinate_file_path, n, coordinates, min_radius, max_radius, angle)
    top_coordinates_list = [coordinate for coordinate, _, _ in top_N_coordinates]

    # Print the results
    print("Top N coordinates with most points in any direction:")
    for coordinate, direction, point_count in top_N_coordinates:
        print(f"coordinate: {coordinate}, Best Direction: {direction}, Points: {point_count}")
