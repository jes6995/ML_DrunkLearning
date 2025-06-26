import moviepy.video.io.VideoFileClip as mp
import os

def remove_audio_from_video(video_path, output_path):
    try:
        # Load the video file
        video = mp.VideoFileClip(video_path)

        # Remove the audio
        video_without_audio = video.without_audio()

        # Save the video without audio to the specified location
        video_without_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"Audio removed from video. File saved at: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Provide the path to the folder containing video files
    folder_path = input("Enter the path to the folder containing video files: ")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print("No video files found in the specified folder.")
        else:
            for video_file in video_files:
                video_path = os.path.join(folder_path, video_file)
                output_path = os.path.join(folder_path, os.path.splitext(video_file)[0] + "_silent" + os.path.splitext(video_file)[1])
                remove_audio_from_video(video_path, output_path)
    else:
        print("The specified folder does not exist or is not a directory.")
