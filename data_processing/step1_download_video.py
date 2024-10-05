import os
import subprocess
import time

def download_video(youtube_link, output_path, cookies_path="cookies.txt"):
    start_time = time.time()
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
        print(f"Video downloaded: {output_path}")
        return output_path
    else:
        print(f"Error downloading video: {result.stderr}")
        return None

youtube_link = "https://www.youtube.com/watch?v=rkfFCSbWDyY"
outpath_folder = "output"
video_id = youtube_link.split('=')[-1]
output_path = os.path.join(outpath_folder, f"{video_id}.mp4")
downloaded_video_path = download_video(youtube_link, output_path)
