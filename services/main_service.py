# services/main_service.py

import os
from utils.video_download import download_video
from utils.frame_extractor import extract_frames_opencv
import config.config as cf
from services.pixtral_service import process_with_pixtral
from services.embedding_service import embed_frame_texts

def main_service(youtube_link, video_dir_path, frame_number, dir_path):
    """
    Main service to download video, extract frames, process with Pixtral, and embed frame texts.

    Parameters:
    youtube_link (str): URL of the YouTube video.
    video_dir_path (str): Path to save the downloaded video.
    frame_number (int): Number of frames to extract.
    dir_path (str): Directory path for saving outputs.

    Returns:
    bool: True if the process completes successfully.
    """
    try:
        description_output_path = os.path.join(dir_path, "video_description.txt")

        frame_folder_path = os.path.join(dir_path, "frames")
        os.makedirs(frame_folder_path, exist_ok=True)

        pixtral_response_path = os.path.join(dir_path, "pixtral_response")
        os.makedirs(pixtral_response_path, exist_ok=True)

        if not os.path.exists(video_dir_path):
            download_video(youtube_link, video_dir_path, description_output_path)
        else:
            print("Video exists in folder.")

        if len(os.listdir(frame_folder_path)) != int(frame_number):
            frames = extract_frames_opencv(
                video_path=video_dir_path, 
                max_frames=frame_number, 
                trim_start=cf.TRIM_START_FRAMES,
                trim_end=cf.TRIM_END_FRAMES, 
                frame_folder=frame_folder_path
            )
        else:
            print("Frames exist in folder.")
        
        # Call the Pixtral service function
        image_files = sorted([f for f in os.listdir(frame_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        process_with_pixtral(image_files, frame_folder_path, pixtral_response_path)

        # Construct index_save_path using INDEX_PATH from config
        # Assuming each video has a unique directory under INDEX_PATH
        # Example: data/video_id/vector_index
        video_id = youtube_link.split('=')[-1]
        index_save_dir = os.path.join(cf.INDEX_PATH, video_id)
        os.makedirs(index_save_dir, exist_ok=True)
        index_save_path = os.path.join(index_save_dir, "vector_index")

        # Embed the Pixtral responses and create vector store index
        embed_frame_texts(frames_folder_path=pixtral_response_path, index_save_path=index_save_path)

        return True

    except Exception as e:
        print(f"An error occurred in main_service: {e}")
        return False
