import json
import numpy as np
import pickle
from sklearn.svm import OneClassSVM

def process_3d_list(data):
    """
    Processes a list of videos (3D array) into a 2D array with padding to handle varying frame sizes.
    """
    try:
        # Determine the maximum frame size across all videos after combining left and right frames
        max_features = max(
            #len(video[0]) + len(video[1]) for video in data
            1636,5
        )

        # Combine and pad each video
        combined_frames = []
        for video in data:
            left_frame = video[0]
            right_frame = video[1]

            combined_frame = np.concatenate([left_frame, right_frame])

            padded_frame = np.pad(combined_frame, (0, max_features - len(combined_frame)), mode="constant")
            combined_frames.append(padded_frame)

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

    deviation_data = process_3d_list(load_json(deviation_file))
    rate_of_change_data = process_3d_list(load_json(rate_of_change_file))
    normalized_data = process_3d_list(load_json(normalized_file))

    if not (deviation_data.shape[0] == rate_of_change_data.shape[0] == normalized_data.shape[0]):
        raise ValueError("Mismatched number of samples across datasets.")

    features = np.column_stack([rate_of_change_data])
    print(f"Combined feature matrix shape: {features.shape}")
    return features

def detect_anomalies(features, model_path="one_class_svm_model.pkl"):
    """
    Loads a pre-trained One-Class SVM model and detects anomalies in the given data.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None

    predictions = model.predict(features)
    return predictions

if __name__ == "__main__":
    # Paths to the JSON files
    folder = input("Enter the folder path to saved data (Make sure it is xxx_data.json): ")

    if len(folder) < 1:
        folder = r'C:\1. School\CMPSC 441\ML_DrinkingModel\Scripts\drunk_data'


    deviation_file = folder + r"\deviation_data.json"
    rate_of_change_file = folder + r"\rate_change_data.json"
    normalized_file = folder + r"\normalized_data.json"

    try:
        # Load the test data
        features = load_data(deviation_file, rate_of_change_file, normalized_file)

        if features is not None:
            # Detect anomalies using the trained One-Class SVM model
            anomaly_predictions = detect_anomalies(features)

            if anomaly_predictions is not None:
                # Output results
                anomaly_count = np.sum(anomaly_predictions == -1)  # Count the anomalies
                for i, prediction in enumerate(anomaly_predictions):
                    status = "Anomaly" if prediction == -1 else "Normal"
                    print(f"Sample {i}: {status}")

                # Correctly formatted summary
                anomalous_indices = np.where(anomaly_predictions == -1)[0]
                print(f"Anomalous samples detected at indices: {anomalous_indices}")
                print(f"\n\nTotal anomalies: {anomaly_count} out of {len(anomaly_predictions)}")


    except Exception as e:
        print(f"Error occurred: {e}")
