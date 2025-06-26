import numpy as np

def process_frame(face_landmarks, frame):
    height, width, _ = frame.shape

    # Get pupil landmarks
    left_pupil = face_landmarks.landmark[468]
    right_pupil = face_landmarks.landmark[473]

    # Define landmarks for static face regions (excluding eyes and mouth)
    static_landmarks = [
        face_landmarks.landmark[168],  # Base nose
        #face_landmarks.landmark[8],    # Between eyes
        #face_landmarks.landmark[9],    # Between eyes 2
        #face_landmarks.landmark[151],  # Forehead
        #face_landmarks.landmark[199],  # Chin

        # Add more landmarks as needed
    ]

    # Calculate average position of static landmarks
    static_x = np.mean([landmark.x for landmark in static_landmarks]) * width
    static_y = np.mean([landmark.y for landmark in static_landmarks]) * height

    # Calculate pixel coordinates of pupils
    left_pupil_x = int(left_pupil.x * width)
    left_pupil_y = int(left_pupil.y * height)
    right_pupil_x = int(right_pupil.x * width)
    right_pupil_y = int(right_pupil.y * height)

    # Calculate relative pupil positions with respect to the average static position
    left_pupil_relative_x = int(left_pupil_x - static_x)
    left_pupil_relative_y = int(left_pupil_y - static_y)
    right_pupil_relative_x = int(right_pupil_x - static_x)
    right_pupil_relative_y = int(right_pupil_y - static_y)

    left_iris_mark = face_landmarks.landmark[475]
    right_iris_mark = face_landmarks.landmark[477]

    left_iris = (int(left_iris_mark.x * width), int(left_iris_mark.y * height))
    right_iris = (int(right_iris_mark.x * width), int(right_iris_mark.y * height))

    return (
         left_pupil_relative_x, right_pupil_relative_x,
         (left_pupil_relative_x + int(static_x), left_pupil_relative_y + int(static_y)),
         (right_pupil_relative_x + int(static_x), right_pupil_relative_y + int(static_y))
    )



def save():
    return
   # return (
   #     left_pupil_relative_x, right_pupil_relative_x,
   #     (left_pupil_relative_x + int(static_x), left_pupil_relative_y + int(static_y)),
   #     (right_pupil_relative_x + int(static_x), right_pupil_relative_y + int(static_y))
   # )



