import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             auc, precision_recall_curve, average_precision_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, Callback

# --- Configuration ---
SEQUENCE_LENGTH = 3
N_SPLITS = 5
RANDOM_STATE = 42
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 10

def compute_features(array):
    """Computes features for a given eye data array."""
    array = np.array(array, dtype=float)  # Ensure correct dtype
    features = {
        'mean': np.mean(array),
        'std': np.std(array),
        'max': np.max(array),
        'min': np.min(array),
        'range': np.max(array) - np.min(array),
        'skewness': pd.Series(array).skew(),
        'kurtosis': pd.Series(array).kurt()
    }
    diff = np.diff(array)
    features.update({
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'max_diff': np.max(diff),
        'min_diff': np.min(diff),
        'range_diff': np.max(diff) - np.min(diff)
    })
    return features

def process_csv(file_path, label):
    """Processes a single CSV file and extracts features."""
    try:
        data = pd.read_csv(file_path, header=None)
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")

    features_list = []
    for _, row in data.iterrows():
        try:
            left_eye_cleaned = np.array(row[0].strip('[]').split(), dtype=float)
            right_eye_cleaned = np.array(row[1].strip('[]').split(), dtype=float)
        except Exception as e:
            raise ValueError(f"Error processing row: {row}\nDetails: {e}")

        left_features = compute_features(left_eye_cleaned)
        right_features = compute_features(right_eye_cleaned)

        features = {f'left_{key}': value for key, value in left_features.items()}
        features.update({f'right_{key}': value for key, value in right_features.items()})
        features['label'] = label
        features_list.append(features)
    return features_list

def process_all_data(data_dir):
    """Processes all data files from the given directory."""
    file_names = [
        "rate_change_data.csv", "rate_change_data_drunk.csv",
        "normalized_data.csv", "normalized_data_drunk.csv",
        "deviation_data.csv", "deviation_data_drunk.csv"
    ]
    sober_files = [os.path.join(data_dir, name) for name in file_names[::2]]
    drunk_files = [os.path.join(data_dir, name) for name in file_names[1::2]]

    data = []
    for sober_file, drunk_file in zip(sober_files, drunk_files):
        if not os.path.exists(sober_file) or not os.path.exists(drunk_file):
            raise FileNotFoundError(f"One of the required files is missing: {sober_file} or {drunk_file}")

        data.extend(process_csv(sober_file, label=0))
        data.extend(process_csv(drunk_file, label=1))
    return data

class AccuracyLogger(Callback):
    """Custom callback to print accuracy at each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy', 0.0)
        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}")

def create_model(input_shape):
    """Creates and compiles the RNN model."""
    model = tf.keras.Sequential([
        layers.Bidirectional(layers.SimpleRNN(128, activation='relu', input_shape=input_shape, return_sequences=True)),
        layers.Dropout(0.2),
        layers.SimpleRNN(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(X_train_rnn, y_train, X_val_rnn, y_val):
    """Trains and evaluates the model for a single fold."""
    model = create_model(input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(X_train_rnn, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val_rnn, y_val), verbose=0,
              callbacks=[early_stopping, AccuracyLogger()])
    _, val_acc = model.evaluate(X_val_rnn, y_val, verbose=0)
    y_pred = (model.predict(X_val_rnn) > 0.5).astype(int).flatten()
    return val_acc, y_pred, model

def visualize_groupings(X_scaled, y):
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Alternatively, you can use t-SNE for non-linear dimensionality reduction
    # tsne = TSNE(n_components=2, random_state=42)
    # X_pca = tsne.fit_transform(X_scaled)

    # Plot the 2D representation
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                    c=color, label="Sober" if label == 0 else "Drunk", alpha=0.6)

    plt.title("Visualization of Drunk vs. Sober Groupings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_dir = input("Enter the directory containing the CSV files: ")
    data = process_all_data(data_dir)
    dataset = pd.DataFrame(data)
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples = (X_scaled.shape[0] // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
    X_rnn = X_scaled[:n_samples].reshape(-1, SEQUENCE_LENGTH, X_scaled.shape[1])

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    conf_matrices = []
    y_true_all, y_pred_all = [], []

    model = None
    for train_index, val_index in kf.split(X_rnn):
        X_train_rnn, X_val_rnn = X_rnn[train_index], X_rnn[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        val_acc, y_pred, model = train_and_evaluate_model(X_train_rnn, y_train, X_val_rnn, y_val)
        accuracies.append(val_acc)
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        conf_matrices.append(confusion_matrix(y_val, y_pred))

    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler after training
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    avg_accuracy = np.mean(accuracies)
    print(f"Average Validation Accuracy: {avg_accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_true_all, y_pred_all))

    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true_all, y_pred_all)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision_score(y_true_all, y_pred_all):.2f})')
    plt.legend()
    plt.show()

    avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
    sns.heatmap(avg_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Sober', 'Drunk'], yticklabels=['Sober', 'Drunk'])
    plt.show()

