import random

def divide_and_sample_with_coordinates(file_path, divisions, samples_per_region, origin, resolution):
    """
    Evenly divide the occupancy grid, sample a specified number of points from each division, 
    and convert the selected indices into actual coordinates. 
    Coordinates are rounded to two decimal places.
    
    :param divisions: (row_divisions, col_divisions) Number of divisions for rows and columns
    :param samples_per_region: Number of samples to extract from each divided region
    :param origin: (origin_row, origin_col) Origin index in the grid
    :param resolution: Actual distance between indices
    :return: List of selected indices and their converted real-world coordinates
    """

    # Create an empty list to store data
    grid = []

    # Read the file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the line and convert characters to integers
            row = [int(cell) for cell in line.strip()]
            grid.append(row)
    
    rows, cols = len(grid), len(grid[0])
    row_divisions, col_divisions = divisions
    region_height = rows // row_divisions
    region_width = cols // col_divisions
    
    selected_indices = []
    coordinates = []  # List to store real-world coordinates
    
    # Sample from each divided region
    for r in range(row_divisions):
        for c in range(col_divisions):
            region_indices = [
                (row, col)
                for row in range(r * region_height, (r + 1) * region_height)
                for col in range(c * region_width, (c + 1) * region_width)
                if grid[row][col] == 0  # Include only indices with a value of 0
            ]
            # Randomly select the required number of samples from the region
            if len(region_indices) >= samples_per_region:
                selected = random.sample(region_indices, samples_per_region)
                selected_indices.extend(selected)
            else:
                selected_indices.extend(region_indices)
    
    # Convert indices to real-world coordinates
    origin_row, origin_col = origin
    for idx in selected_indices:
        row, col = idx
        x = round((col - origin_col) * resolution, 2)  # Calculate X coordinate, rounded to two decimals
        y = round((origin_row - row) * resolution, 2)  # Calculate Y coordinate, rounded to two decimals
        coordinates.append((x, y))

    return coordinates

if __name__ == "__main__":

    # File reading and processing code
    file_path = "/home/air/open_sw/occupancyGridMap.txt"  # File path

    # Execute grid division and sampling
    divisions = (1, 1)  # Divide into 4 regions for rows and columns respectively
    samples_per_region = 100  # Select 2 samples from each region
    origin = (52, 52)  # Origin index in the grid
    resolution = 0.05  # Actual distance between indices
    
    coordinates = divide_and_sample_with_coordinates(file_path, divisions, samples_per_region, origin, resolution)
    
    # Print the results
    print("Selected indices and real-world coordinates:")
    for coord in coordinates:
        print(f"Coordinate: {coord}")
