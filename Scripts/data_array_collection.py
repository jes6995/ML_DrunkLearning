import cv2  # pip install opencv-python
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Initialize face mesh solution
mp_face_mesh = mp.solutions.face_mesh


def process_frame(face_landmarks, frame):
    height, width, _ = frame.shape

    # Get pupil landmarks
    left_pupil = face_landmarks.landmark[468]
    right_pupil = face_landmarks.landmark[473]

    # Convert normalized landmark coordinates to pixel coordinates
    left_pupil_x = int(left_pupil.x * width)
    right_pupil_x = int(right_pupil.x * width)
    left_pupil_y = int(left_pupil.y * height)
    right_pupil_y = int(right_pupil.y * height)

    # Get eye corner landmarks
    left_eye_inner_corner_x = int(face_landmarks.landmark[133].x * width)
    left_eye_outer_corner_x = int(face_landmarks.landmark[33].x * width)
    right_eye_inner_corner_x = int(face_landmarks.landmark[362].x * width)
    right_eye_outer_corner_x = int(face_landmarks.landmark[263].x * width)

    # Calculate eye socket widths
    left_eye_socket_width = abs(left_eye_inner_corner_x - left_eye_outer_corner_x)
    right_eye_socket_width = abs(right_eye_outer_corner_x - right_eye_inner_corner_x)

    # Calculate relative pupil positions
    left_pupil_relative_x = (left_pupil_x - left_eye_outer_corner_x) / left_eye_socket_width
    right_pupil_relative_x = (right_pupil_x - right_eye_inner_corner_x) / right_eye_socket_width

    return left_pupil_relative_x, right_pupil_relative_x, (left_pupil_x, left_pupil_y), (right_pupil_x, right_pupil_y)


def plot_pupil_positions(left_pupil_normalized, right_pupil_normalized, frame_count):
    plt.figure(figsize=(10, 6))
    plt.plot(range(frame_count), left_pupil_normalized, label='Left Pupil', color='blue')
    plt.plot(range(frame_count), right_pupil_normalized, label='Right Pupil', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Pupil X-Position (Normalized)')
    plt.title('Normalized Pupil X-Position Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pupil_positions_ma(left_pupil_positions_norm, right_pupil_positions_norm, frame_count, window_size=5):
    """Plots the normalized pupil positions with moving average over time."""

    # Calculate moving averages
    left_pupil_ma = np.convolve(left_pupil_positions_norm, np.ones(window_size) / window_size, mode='valid')
    right_pupil_ma = np.convolve(right_pupil_positions_norm, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 6))

    # Plot moving averages
    plt.plot(range(window_size, frame_count + 1), left_pupil_ma, label='Left Pupil MA', color='blue')
    plt.plot(range(window_size, frame_count + 1), right_pupil_ma, label='Right Pupil MA', color='red')

    plt.xlabel('Frame Number')
    plt.ylabel('Pupil X-Position (Normalized)')
    plt.title(f'Normalized Pupil X-Position with Moving Average ({window_size} Frames) Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_rate_of_change(pupil_normalized):
    return np.gradient(pupil_normalized)


def plot_rate_of_change(left_pupil_rate_of_change, right_pupil_rate_of_change, frame_count):
    plt.figure(figsize=(10, 6))
    plt.plot(range(frame_count), left_pupil_rate_of_change, label='Left Pupil', color='blue')
    plt.plot(range(frame_count), right_pupil_rate_of_change, label='Right Pupil', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Rate of Change of Pupil X-Position')
    plt.title('Rate of Change of Normalized Pupil X-Position Over Time')
    plt.ylim(-.1, .1)
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_deviations(rate_of_change, window_size, std_multiplier):
    avg_rate_of_change = np.convolve(rate_of_change, np.ones(window_size) / window_size, mode='valid')
    std_dev = np.std(avg_rate_of_change)
    threshold = std_multiplier * std_dev
    deviations = np.where(np.abs(rate_of_change[window_size:]) > threshold, rate_of_change[window_size:], 0)
    return deviations


def plot_deviations(left_deviations, right_deviations, window_size, frame_count):
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size, frame_count), left_deviations, label='Left Pupil', color='blue')
    plt.plot(range(window_size, frame_count), right_deviations, label='Right Pupil', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Significant Deviations in Rate of Change')
    plt.title(f'Significant Deviations from Moving Average ({window_size} Frames) of Pupil X-Position')
    plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.grid(True)
    plt.show()


def process_video(video_path, window_size_roc, window_size_ma, std_multiplier=2.0):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    left_pupil_relative_x_positions = []
    right_pupil_relative_x_positions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frameHeight = int(frame.shape[0] * 0.3)
        frameWidth = int(frame.shape[1] * 0.3)
        frame = cv2.resize(frame, (frameWidth, frameHeight))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_pupil_relative_x, right_pupil_relative_x, left_pupil, right_pupil = process_frame(face_landmarks, frame)

                left_pupil_relative_x_positions.append(left_pupil_relative_x)
                right_pupil_relative_x_positions.append(right_pupil_relative_x)

                # Draw pupils
                cv2.drawMarker(frame, left_pupil,(0, 0, 255), cv2.MARKER_CROSS, 10)
                cv2.drawMarker(frame, right_pupil, (0, 255, 0), cv2.MARKER_CROSS, 10)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Normalize the pupil lists
    left_pupil_normalized = (np.array(left_pupil_relative_x_positions) - min(left_pupil_relative_x_positions)) / (max(left_pupil_relative_x_positions) - min(left_pupil_relative_x_positions))
    right_pupil_normalized = (np.array(right_pupil_relative_x_positions) - min(right_pupil_relative_x_positions)) / (max(right_pupil_relative_x_positions) - min(right_pupil_relative_x_positions))

    return {
        "left_pupil_normalized": left_pupil_normalized,
        "right_pupil_normalized": right_pupil_normalized,
        "frame_count": frame_count
    }


def main():
    video_paths = [
        r'C:\Users\jared\Documents\IMG_9260.MOV',
        r'C:\Users\jared\Documents\23.MOV',
        r'C:\Users\jared\Documents\miles_sober5.mp4'
        # ... add more paths here ...
    ]
    window_size_roc = int(60 * .1)
    window_size_ma = int(3)
    std_multiplier = 3

    all_video_data = []

    for video_path in video_paths:
        data = process_video(video_path, window_size_roc, window_size_ma, std_multiplier)

        video_frame_data = []
        for i in range(data["frame_count"]):
            video_frame_data.append([
                data["left_pupil_normalized"][i],
                data["right_pupil_normalized"][i]
            ])
        all_video_data.append(video_frame_data)

    # --- Print details of the array ---
    for i, video_data in enumerate(all_video_data):
        print(f"--- Video {i + 1} ---")
        print(f"Number of frames: {len(video_data)}")
        print("Sample frame data:")
        for j in range(min(5, len(video_data))):  # Print data for the first 5 frames
            print(f"  Frame {j + 1}: {video_data[j]}")
        print("...")  # Indicate that there might be more frames

    return all_video_data


if __name__ == "__main__":
    main()