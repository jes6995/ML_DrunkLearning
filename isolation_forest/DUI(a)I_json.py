import os
import json
import numpy as np
from FileLoop import get_file_paths
import video_process

def write_json(file_path, data):
    """Write data to a JSON file, overwriting any existing content."""
    # Convert all numpy arrays in the data to Python lists
    data_serializable = [
        np.array(entry).tolist() if isinstance(entry, np.ndarray) else entry
        for entry in data
    ]
    with open(file_path, mode='w+') as file:
        json.dump(data_serializable, file, indent=4)

def main():
    directory_path = input('Enter your directory path for the video folder: ')  # Replace with your folder path

    deviation_file = r"data/deviation_data_new.json"
    rate_change_file = r"data/rate_change_data_new.json"
    normalized_file = r"data/normalized_data_new.json"

    file_list = get_file_paths(directory_path)

    deviation_data = []  # Initialize lists to collect all data
    rate_change_data = []
    normalized_data = []

    counter = 1

    for x in file_list:

        _, file_ext = os.path.splitext(x)

        # Check if the file has the desired extension (case-insensitive)
        if file_ext.lower() not in ('.mov', '.mp4'):
            continue

        # Process the video file
        deviationVal, rateOfChange, normalizedData = video_process.main(x, False, 2)

        # Collect data
        deviation_data.append(deviationVal)
        rate_change_data.append(rateOfChange)
        normalized_data.append(normalizedData)

        print('####################################\n\nCOUNT ',
              counter, '\n\n#################################')
        print(f"Processed {x}")
        counter += 1

    print('\n\n\nDONE ######################################### DONE\n\n\n')

    # Write all collected data to JSON files
    write_json(deviation_file, deviation_data)
    write_json(rate_change_file, rate_change_data)
    write_json(normalized_file, normalized_data)

if __name__ == "__main__":
    main()
