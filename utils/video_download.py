import os
import subprocess
import time

def download_video(youtube_link, video_output_path, description_output_path, cookies_path="cookies.txt"):
    start_time = time.time()
    try:
        # Download the video
        command_video = [
            "yt-dlp",
            "--cookies", cookies_path,
            "-f", "mp4",
            youtube_link,
            "-o", video_output_path
        ]
        result_video = subprocess.run(command_video, capture_output=True, text=True)

        if result_video.returncode == 0:
            print(f"Video downloaded successfully: {video_output_path}")
        else:
            print(f"Error downloading video: {result_video.stderr}")
            return

        # Get and save the video description
        command_description = [
            "yt-dlp",
            "--cookies", cookies_path,
            "--get-description",
            youtube_link
        ]
        result_description = subprocess.run(command_description, capture_output=True, text=True)

        if result_description.returncode == 0:
            description = result_description.stdout
            with open(description_output_path, 'w', encoding='utf-8') as file:
                file.write(description)
            print(f"Video description saved to: {description_output_path}")
        else:
            print(f"Error getting video description: {result_description.stderr}")
        
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Total time taken: {time_taken:.2f} seconds")
    
    except FileNotFoundError as fnf_error:
        print(f"Error: yt-dlp not found. Make sure it's installed. Details: {fnf_error}")
    except subprocess.CalledProcessError as cpe:
        print(f"Subprocess error: {cpe}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
