import pickle
import numpy as np
import json
import isolation_forest_training


def process_3d_list(data):
    """
    Processes a list of videos (3D array) into a 2D array with padding to handle varying frame sizes.

    Args:
        data: List of videos, where each video is a list containing two arrays (left and right features),
              and each frame is a list of features.

    Returns:
        A 2D NumPy array (samples x features) with padding to equalize frame sizes.
    """
    try:
        # Determine the maximum frame size across all videos after combining left and right frames
        # The number of features must be the same as the training ones
        max_features = max(
            #len(video[0]) + len(video[1]) for video in data
            1636, 5
        )

        # Combine and pad each video
        combined_frames = []
        for video in data:
            # Extract left and right frames
            left_frame = video[0]
            right_frame = video[1]

            # Combine left and right frames
            combined_frame = np.concatenate([left_frame, right_frame])

            # Pad the combined frame to the maximum size
            padded_frame = np.pad(
                combined_frame, (0, max_features - len(combined_frame)), mode="constant"
            )
            combined_frames.append(padded_frame)

        # Convert to NumPy array
        result = np.array(combined_frames)
        print(f"Processed data to shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

def load_data(deviation_file, rate_of_change_file, normalized_file, choice):
    """
    Loads and processes 3D nested JSON data into a consistent NumPy array format.
    Each JSON file should contain an array of videos, where each video is an array of frames, and each frame has features.
    """
    # Load JSON data
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

    # Ensure all arrays have the same number of samples (frames)
    if not (
        deviation_data.shape[0]
        == rate_of_change_data.shape[0]
        == normalized_data.shape[0]
    ):
        raise ValueError(
            "Mismatched number of samples across datasets. Ensure all files represent the same data."
        )

    # Combine data into a single feature matrix
    #features = np.column_stack([deviation_data, rate_of_change_data, normalized_data])

    if choice == '1':
        print('Feature is choice 1')
        features = np.column_stack([deviation_data])
    elif choice == '2':
        print('Feature is choice 2')
        features = np.column_stack([rate_of_change_data])
    else:
        print('Feature is choice 3')
        features = np.column_stack([normalized_data])

    print(f"Combined feature matrix shape: {features.shape}")
    return features


def predict_anomalies(features, model_path="anomaly_model.pkl"):
    """
    Predicts whether the given features are anomalies using a pre-trained model.

    Args:
        features: A NumPy array of shape (samples, features) containing the data to analyze.
        model_path: Path to the saved model.

    Returns:
        A NumPy array of predictions (1 for normal, -1 for anomaly).
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            print(f"Model loaded from {model_path}.")
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None

    # Predict anomalies
    predictions = model.predict(features)
    return predictions


if __name__ == "__main__":
    # Paths to your JSON files
    folder = input("Enter the folder path to saved data (Make sure it is xxx_data.json): ")

    if len(folder) < 1:
        folder = r'C:\1. School\CMPSC 441\ML_DrinkingModel\Scripts\drunk_data'

    deviation_file = folder + r"\deviation_data.json"
    rate_of_change_file = folder + r"\rate_change_data.json"
    normalized_file = folder + r"\normalized_data.json"

    # Path to the trained model
    model_path = input("Enter the path to the saved model: ")
    if len(model_path) < 1:
        model_path = r'/Scripts/anomaly_model.pkl'

    choice = input("Do you want 1: deviation, 2: rate of change, 3: normalization? ")

    try:
        # Load features using the load_data function
        features = load_data(deviation_file, rate_of_change_file, normalized_file, choice=choice)

        # Check if features are successfully loaded
        if features is not None:
            print("Features loaded successfully.")

            # Predict anomalies
            predictions = predict_anomalies(features, model_path)

            if predictions is not None:
                # Output results
                anomaly_count = np.sum(predictions == -1)  # Count the anomalies
                for i, prediction in enumerate(predictions):
                    status = "Anomaly" if prediction == -1 else "Normal"
                    print(f"Sample {i}: {status}")

                # Correctly formatted summary
                anomalous_indices = np.where(predictions == -1)[0]
                print(f"\n\nAnomalous samples detected at indices: {anomalous_indices}")
                print(f"Total anomalies: {anomaly_count} out of {len(predictions)}")


            else:
                print("Error: Could not predict anomalies.")
    except Exception as e:
        print(f"Error occurred: {e}")
