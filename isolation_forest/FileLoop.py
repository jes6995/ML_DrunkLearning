import os

def get_file_paths(directory):
    """
    Collects all file paths in the given directory and its subdirectories.

    Args:
        directory (str): The directory to search for files.

    Returns:
        list: A list of full file paths.
    """
    file_paths = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file)) 
    return file_paths
