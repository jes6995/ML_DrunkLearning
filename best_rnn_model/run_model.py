import os
import pickle  # Import pickle for loading the model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import dui
import train_model

def clear_csv_files(file_paths):
    """Clears the contents of the specified CSV files."""
    for file_path in file_paths:
        try:
            with open(file_path, 'w') as f:
                f.truncate()  # Clear the file
                f.close()
            print(f"Cleared contents of file: {file_path}")
        except Exception as e:
            print(f"Error clearing file {file_path}: {e}")

def process_all_data(data_dir):
    """Processes all data files from the given directory."""
    file_names = [
        "rate_change_data_new.csv",
        "normalized_data_new.csv",
        "deviation_data_new.csv",
    ]
    files = [os.path.join(data_dir, name) for name in file_names]

    data = []
    for file in files:  # Fixed incorrect zip usage
        data.extend(train_model.process_csv(file, label=0))  # Corrected index usage
    return data



def main():
    SEQUENCE_LENGTH = 3

    # Get data directory from user input
    #data_dir = input("\n\nPlease enter the path the data folder inside the final folder (No quotes allowed): ")

    # This should be the path to the data folder in best_rnn_model
    data_dir = os.path.dirname(os.path.abspath(__file__)) + r'\data'

    # Specify file paths
    file_names = [
        "rate_change_data_new.csv",
        "normalized_data_new.csv",
        "deviation_data_new.csv",
    ]
    file_paths = [os.path.join(data_dir, name) for name in file_names]

    # Clear contents of CSV files
    clear_csv_files(file_paths)

    # This should be the path to a folder containing test videos. No quotes allowed.
    video_folder = input("\nPlease enter the path to the folder holding the video (No quotes allowed): ")
    dui.main(video_folder, False)

    # Process data
    data = process_all_data(data_dir)
    if not data:
        print("\nError: No data found. Please check the input files.")
        return

    dataset = pd.DataFrame(data)
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    scaler = None
    # Load scaler during prediction
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)
    X_rnn = X_scaled[:(X_scaled.shape[0] // SEQUENCE_LENGTH) * SEQUENCE_LENGTH].reshape(
        (-1, SEQUENCE_LENGTH, X_scaled.shape[1]))

    # --- Load the saved model ---
    try:
        with open("trained_model.pkl", "rb") as f:
            model = pickle.load(f)

    except FileNotFoundError:
        print("Error: Trained model file 'trained_model.pkl' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Run prediction ---
    try:
        print('\n\nX_rnn Shape:', X_rnn.shape)

        model.training = False
        predictions = model.predict(X_rnn)

        # Debugging: Print predictions array
        print('\nRaw Predictions:', predictions)

        # Handle different output formats
        if len(predictions.shape) == 1:  # 1-D output
            predicted_classes = ["Drunk" if pred > 0.5 else "Sober" for pred in predictions]
        elif len(predictions.shape) == 2 and predictions.shape[1] == 1:  # Column vector
            predicted_classes = ["Drunk" if pred[0] > 0.5 else "Sober" for pred in predictions]
        else:
            print("\nUnexpected prediction shape:", predictions.shape)
            return

        print("\nPredictions:", predicted_classes)
    except Exception as e:
        print(f"\nError during prediction: {e}")

if __name__ == '__main__':
    main()
