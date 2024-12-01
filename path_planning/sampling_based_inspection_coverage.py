import plyToTxt as ptt
import samplingPoint as sp
import sectorPointCheck as spc
import tsp

def path_finding():
    
    ply_file_path = '/home/air/open_sw/ply/opensw.ply'   # Path to the input PLY file
    coordinates_file_path = '/home/air/open_sw/coordinates_osw.txt'  # Path to save the extracted and sorted coordinates
    occupancy_file_path = "/home/air/open_sw/occupancyGridMap_osw.txt"  # File path for occupancy grid map

    # Convert PLY file to a text file with coordinates
    ptt.plyToTxt(ply_file_path, coordinates_file_path)

    # Execute grid division and sampling
    divisions = (1, 1)  # Divide the grid into 4 regions for rows and columns respectively
    samples_per_region = 100  # Select 2 samples from each region
    origin = (26, 14)  # Origin index in the grid
    resolution = 0.05  # Actual distance between indices

    coordinates = sp.divide_and_sample_with_coordinates(occupancy_file_path, divisions, samples_per_region, origin, resolution)

    n = 10  # Number of top coordinates to retrieve
    start_point = (0.0, 0.0)  # Starting point
    min_radius = 0.5  # Minimum radius of the sector
    max_radius = 1.25  # Maximum radius of the sector
    angle = 60  # Angle of the sector in degrees

    # Find the top N coordinates
    top_N_coordinates = spc.find_top_N_coordinates(coordinates_file_path, n, coordinates, min_radius, max_radius, angle)
    top_N_coordinates.insert(0, (start_point, (1, 0), 0))
    top_coordinates_list = [coordinate for coordinate, _, _ in top_N_coordinates]
    top_directions_list = [direction for _, direction, _ in top_N_coordinates]

    # Find the optimal route and its cost using the Held-Karp algorithm
    optimal_route, min_cost = tsp.tsp_held_karp(top_coordinates_list)

    # Append the coordinates and directions in the order specified by the optimal route
    print(f"Optimal Route: {optimal_route}")
    print(f"Total Cost: {min_cost:.2f}")

    coordinates_result = []
    directions_result = []

    # todo
    for i in optimal_route:
        coordinates_result.append(top_coordinates_list[i])
        directions_result.append(top_directions_list[i])

    print("generate route!")
    return coordinates_result, directions_result
    
if __name__ == "__main__":
    coordinates_result, directions_result = path_finding()
    
    for coordinate, direction in zip(coordinates_result, directions_result):
        print(f"{coordinate}, {direction}")