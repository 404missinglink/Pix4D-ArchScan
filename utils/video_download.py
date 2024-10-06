import os
import subprocess
import time

def download_video(youtube_link, output_path, cookies_path="cookies.txt"):
    start_time = time.time()
    try:
        # Run the yt-dlp command
        command = [
            "yt-dlp",
            "--cookies", cookies_path,
            "-f", "mp4",
            youtube_link,
            "-o", output_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken to download the video: {time_taken:.2f} seconds")
        
        if result.returncode == 0:
            print(f"Video downloaded successfully: {output_path}")
        else:
            print(f"Error downloading video: {result.stderr}")
    except FileNotFoundError as fnf_error:
        print(f"Error: yt-dlp not found. Make sure it's installed. Details: {fnf_error}")
    except subprocess.CalledProcessError as cpe:
        print(f"Subprocess error: {cpe}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
