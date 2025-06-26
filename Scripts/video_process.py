import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Initialize face mesh solution
mp_face_mesh = mp.solutions.face_mesh


def call_plot_pupil_positions(left_pupil_normalized, right_pupil_normalized, frame_count):

    plt.figure(figsize=(10, 6))
    plt.plot(range(frame_count), left_pupil_normalized, label='Left Pupil', color='blue')
    plt.plot(range(frame_count), right_pupil_normalized, label='Right Pupil', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Pupil X-Position (Normalized)')
    plt.title('Normalized Pupil X-Position Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def call_plot_pupil_positions_ma(left_pupil_ma, right_pupil_ma, frame_count, window_size):
    """Plots the normalized pupil positions with moving average over time."""

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
    avg_rate_of_change = np.convolve(rate_of_change, np.ones(window_size)/window_size, mode='valid')
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


def video_processor(video_path, window_size_roc, window_size_ma, std_multiplier, do_draw, choice):

    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
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

                (left_pupil_relative_x,
                 right_pupil_relative_x,
                 left_pupil, right_pupil) = call_process_frame(face_landmarks, frame, choice)

                left_pupil_relative_x_positions.append(left_pupil_relative_x)
                right_pupil_relative_x_positions.append(right_pupil_relative_x)

                # Draw pupils
                if do_draw:
                    cv2.drawMarker(frame, left_pupil, (0, 0, 255), cv2.MARKER_CROSS, 10)
                    cv2.drawMarker(frame, right_pupil, (0, 255, 0), cv2.MARKER_CROSS, 10)

        if do_draw:
            cv2.imshow('Pupil Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') & do_draw:
            break

        frame_count += 1

    if do_draw:
        cap.release()
        cv2.destroyAllWindows()

        # Normalize the pupil lists
    left_pupil_normalized = (np.array(left_pupil_relative_x_positions) - min(left_pupil_relative_x_positions)) / (
                max(left_pupil_relative_x_positions) - min(left_pupil_relative_x_positions))
    right_pupil_normalized = (np.array(right_pupil_relative_x_positions) - min(right_pupil_relative_x_positions)) / (
                max(right_pupil_relative_x_positions) - min(right_pupil_relative_x_positions))

    # Calculate moving averages
    left_pupil_ma = np.convolve(left_pupil_normalized, np.ones(window_size_ma) / window_size_ma, mode='valid')
    right_pupil_ma = np.convolve(right_pupil_normalized, np.ones(window_size_ma) / window_size_ma, mode='valid')

    # Plot pupil positions
    if do_draw:
        call_plot_pupil_positions(left_pupil_normalized, right_pupil_normalized, frame_count)

        # Plot pupil positions with moving average
        call_plot_pupil_positions_ma(left_pupil_ma, right_pupil_ma,
                                frame_count=frame_count,
                                window_size=window_size_ma)

    # Calculate rate of change
    left_pupil_rate_of_change = calculate_rate_of_change(left_pupil_normalized)
    right_pupil_rate_of_change = calculate_rate_of_change(right_pupil_normalized)

    # Plot rate of change
    if do_draw:
        plot_rate_of_change(left_pupil_rate_of_change, right_pupil_rate_of_change, frame_count)

    # Calculate deviations
    left_deviations = calculate_deviations(left_pupil_rate_of_change, window_size_roc, std_multiplier)
    right_deviations = calculate_deviations(right_pupil_rate_of_change, window_size_roc, std_multiplier)

    if do_draw:
        # Plot deviations
        plot_deviations(left_deviations, right_deviations, window_size_roc, frame_count)

    deviation_data = [left_deviations, right_deviations]
    rate_of_change_data = [left_pupil_rate_of_change, right_pupil_rate_of_change]
    normalized_data = [left_pupil_normalized, right_pupil_normalized]

    return deviation_data, rate_of_change_data, normalized_data


# Handles the different eye detection algs
def call_process_frame(face_landmarks, frame, choice):

    if choice == 1:
        from mediapip_cv2_socket_anchor import process_frame
    elif choice == 2:
        from mediapipe_raw import process_frame
    elif choice >= 3:
        from mediapipe_anchor_test import process_frame

    # This returns: left_pupil_relative_x, right_pupil_relative_x, (left_pupil_x, left_pupil_y), (right_pupil_x, right_pupil_y)
    return process_frame(face_landmarks, frame)


def main(video_path = '',
         do_draw = True,
         choice = 1):

    if len(video_path) < 1:
        video_path = input('Enter the path to the video: ')

    deviation_data, rate_of_change_data, normalized_data = video_processor(
        video_path,
        window_size_roc = 6,
        window_size_ma = 3,
        std_multiplier = 3,
        do_draw=do_draw,    # Controls plotting
        choice=choice        # 1: socket anchor, 2: raw, 3: test anchor
    )

    # Making them arrays
    normalized_data = np.array([np.array(normalized_data[0]), np.array(normalized_data[1])])
    rate_of_change_data = np.array([np.array(rate_of_change_data[0]), np.array(rate_of_change_data[1])])
    deviation_data = np.array([np.array(deviation_data[0]), np.array(deviation_data[1])])


    return deviation_data, rate_of_change_data, normalized_data



if __name__ == "__main__":
    main()





