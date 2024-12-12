def plyToTxt(input_file_path, output_file_path):    
    """
    Extracts coordinates from a PLY file, sorts them in ascending order by x, y, z, 
    and saves them as a text file.
    """
    # Extract coordinates from the PLY file
    coordinates = []
    
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
    
        # Find the end of the header and set the starting point for data
        data_start_line = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                data_start_line = i + 1
                break
            
        # Extract coordinate data (x, y, z)
        for line in lines[data_start_line:]:
            values = line.split()
            if len(values) >= 3:  # Ensure at least x, y, z values are present
                x, y = map(float, values[:2])
                z = 0
                coordinates.append((x, y, z))
    
        # Sort coordinates in ascending order by x, y, z
        coordinates.sort(key=lambda coord: (coord[0], coord[1]))
    
        # Save sorted coordinates to a text file
        with open(output_file_path, 'w') as file:
            for coord in coordinates:
                file.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
    
        print(f"Coordinates successfully sorted and saved to {output_file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Path to the input PLY file
    input_file_path = '/home/air/open_sw/ply/map.ply'  # Replace with the correct PLY file path
    # Path to save the extracted and sorted coordinates
    output_file_path = '/home/air/open_sw/coordinates.txt'
    plyToTxt(input_file_path, output_file_path)