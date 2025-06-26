import json
import numpy as np
from sklearn.svm import OneClassSVM
import pickle

def process_3d_list(data):
    """
    Processes a list of videos (3D array) into a 2D array with padding to handle varying frame sizes.
    """
    try:
        # Determine the maximum frame size across all videos after combining left and right frames
        max_features = max(len(video[0]) + len(video[1]) for video in data)

        # Combine and pad each video
        combined_frames = []
        for video in data:
            # Extract left and right frames
            left_frame = video[0]
            right_frame = video[1]

            # Combine left and right frames
            combined_frame = np.concatenate([left_frame, right_frame])

            # Pad the combined frame to the maximum size
            padded_frame = np.pad(combined_frame, (0, max_features - len(combined_frame)), mode="constant")
            combined_frames.append(padded_frame)

        # Convert to NumPy array
        result = np.array(combined_frames)
        print(f"Processed data to shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

def load_data(deviation_file, rate_of_change_file, normalized_file):
    """
    Loads and processes 3D nested JSON data into a consistent NumPy array format.
    """
    def load_json(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                print(f"Loaded {file_path} successfully.")
                return data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {file_path}: {e}")
            raise

    # Load and process each file
    deviation_data = process_3d_list(load_json(deviation_file))
    rate_of_change_data = process_3d_list(load_json(rate_of_change_file))
    normalized_data = process_3d_list(load_json(normalized_file))

    # Ensure all arrays have the same number of samples
    if not (deviation_data.shape[0] == rate_of_change_data.shape[0] == normalized_data.shape[0]):
        raise ValueError("Mismatched number of samples across datasets.")

    # Combine data into a single feature matrix
    features = np.column_stack([rate_of_change_data])

    print(f"Combined feature matrix shape: {features.shape}")
    return features

def train_one_class_svm(features, model_path="one_class_svm_model.pkl"):
    """
    Trains a One-Class SVM model on sober data and saves it to a file.
    """
    model = OneClassSVM(
        kernel="rbf",
        nu=0.05,
        gamma='scale',
    )  # Tune parameters as needed
    model.fit(features)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("One-Class SVM model trained and saved to", model_path)

if __name__ == "__main__":
    # Paths to your JSON files
    folder = input("Enter the folder path to saved data (Make sure it is xxx_data.json): ")

    if len(folder) < 1:
        folder = r'C:\1. School\CMPSC 441\ML_DrinkingModel\Scripts\sober_data'

    deviation_file = folder + r"\deviation_data.json"
    rate_of_change_file = folder + r"\rate_change_data.json"
    normalized_file = folder + r"\normalized_data.json"

    try:
        # Load the data
        features = load_data(deviation_file, rate_of_change_file, normalized_file)

        if features is not None:
            # Train the One-Class SVM model
            train_one_class_svm(features)

    except Exception as e:
        print(f"Error occurred: {e}")

