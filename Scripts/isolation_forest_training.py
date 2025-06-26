import json
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import matplotlib.pyplot as plt


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



def detect_anomalies(features, model_path="anomaly_model.pkl"):
    """
    Loads a pre-trained Isolation Forest model and detects anomalies in the given data.

    Args:
      features: A NumPy array of shape (samples, features) containing the data to analyze.
      model_path: Path to the saved model.

    Returns:
      A NumPy array of predictions (1 for normal, -1 for anomaly).
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None

    predictions = model.predict(features)
    return predictions


def visualize_anomalies(features, anomaly_predictions):
    """
    Visualizes the anomaly detection results.

    Args:
      features: A NumPy array of shape (samples, features) containing the data.
      anomaly_predictions: A NumPy array of predictions (1 for normal, -1 for anomaly).
    """
    plt.figure(figsize=(10, 6))

    # Plot each feature with different colors and highlight anomalies
    for i, label in enumerate(["Deviation", "Rate of Change", "Normalized"]):
        plt.scatter(range(len(features)), features[:, i], label=label, alpha=0.5)
        plt.scatter(
            np.where(anomaly_predictions == -1)[0],
            features[anomaly_predictions == -1, i],
            color="red",
            s=10,  # Smaller marker size for anomalies
            label=f"Anomalies ({label})" if i == 0 else None,  # Avoid duplicate labels
            zorder=10
        )

    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    plt.title("Anomaly Detection Results")
    plt.legend()
    plt.show()


def train_anomaly_model(features,
                        model_path="anomaly_model.pkl",
                        contamination=0.05,
                        n_estimators=100,
                        random_state=42,
                        max_features=1,
                        max_samples=700
                        ):
    """
    Trains an Isolation Forest model and saves it to a file.

    Args:
      features: A NumPy array of shape (samples, features) containing the training data.
      model_path: Path to save the trained model.
      contamination: The proportion of outliers in the data set.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        #max_features=max_features,
        max_samples=max_samples
    )
    model.fit(features)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Anomaly detection model trained and saved to", model_path)


if __name__ == "__main__":
    # Paths to your JSON files
    folder = input("Enter the folder path to saved data (Make sure it is xxx_data.json): ")

    if len(folder) < 1:
        folder = r'C:\1. School\CMPSC 441\ML_DrinkingModel\data_training\training_sober'

    choice = input("Do you want 1: deviation, 2: rate of change, 3: normalization? ")

    deviation_file = folder + r"\deviation_data.json"
    rate_of_change_file = folder + r"\rate_change_data.json"
    normalized_file = folder+ r"\normalized_data.json"

    # Load the data
    try:
        features = load_data(deviation_file, rate_of_change_file, normalized_file, choice=choice)

        if features is not None:
            # Train the anomaly detection model
            train_anomaly_model(
                features,
                contamination=.05,  # default .05
                n_estimators=1000,   # default 100 (1000 worked better)
                max_features=1.0,     # default 1 (not being used currently)
                random_state=42,     # default 42 (100 worked better)
                max_samples=2000
            )

            # Detect anomalies
            anomaly_predictions = detect_anomalies(features)

            if anomaly_predictions is not None:
                # Find and print indices of anomalous samples
                anomalous_indices = np.where(anomaly_predictions == -1)[0]
                print(f"Anomalous samples detected at indices: {anomalous_indices}")
                print(f"Total anomalous samples: {len(anomalous_indices)}")

                # Visualize the results
                visualize_anomalies(features, anomaly_predictions)

    except Exception as e:
        print(f"Error occurred: {e}")









