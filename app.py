from services.main_service import main_service
import config.config as cf
from time import time
import os

if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/watch?v=7Q6DU-AuZyI"
    video_id = youtube_link.split('=')[-1]

    dir_path = os.path.join("data", str(video_id))
    os.makedirs(dir_path, exist_ok=True)

    video_dir_path = os.path.join(cf.video_folder_path, f"{video_id}.mp4")
    frame_number = 50

    main_service(youtube_link, video_dir_path, frame_number, dir_path)

