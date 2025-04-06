import json

def normalize(value, min_value, max_value):
    """ Normalize a value to a [0, 1] scale. """
    return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

def calculate_average_cost(path, json_data_path, alpha=0.7):
    """ Calculate a combined cost metric based on normalized effort and distance, loading data from a JSON file. """
    # Load JSON data from the file
    with open(json_data_path, 'r') as file:
        json_data = json.load(file)
    
    # Initialize variables to find min and max values for effort and distance
    min_effort, max_effort = float('inf'), float('-inf')
    min_distance, max_distance = float('inf'), float('-inf')

    # First pass: determine the ranges of effort and distance
    for waypoint, data in json_data.items():
        if 'average_effort' in data and 'distance' in data:
            min_effort = min(min_effort, data['average_effort'])
            max_effort = max(max_effort, data['average_effort'])
            min_distance = min(min_distance, data['distance'])
            max_distance = max(max_distance, data['distance'])

    total_effort = 0
    total_distance = 0
    count = 0
    
    # Second pass: compute totals from the path using the JSON data
    for waypoint in path:
        if waypoint in json_data:
            data = json_data[waypoint]
            total_effort += data['average_effort']
            total_distance += data['distance']
            count += 1
    
    # Calculate averages and normalized values
    if count > 0:
        average_effort = total_effort / count
        average_distance = total_distance / count
        normalized_effort = normalize(average_effort, min_effort, max_effort)
        normalized_distance = normalize(average_distance, min_distance, max_distance)
        # Calculate the weighted average
        average_cost = alpha * normalized_effort + (1 - alpha) * normalized_distance
    else:
        average_cost = 0
    
    return average_cost
