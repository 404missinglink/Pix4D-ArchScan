import sys
import os
import cv2
from PIL import Image
import logging
from utils.video_download import download_video
from utils.frame_extractor import extract_frames_opencv
import config.config as cf
from services.pixtral_service import process_with_pixtral

def main_service(youtube_link, video_dir_path, frame_number, dir_path):

    frame_folder_path = os.path.join(dir_path, "frames")
    os.makedirs(frame_folder_path, exist_ok=True)

    pixtral_response_path = os.path.join(dir_path, "pixtral_response")
    os.makedirs(pixtral_response_path, exist_ok=True)

    if not os.path.exists(video_dir_path):
        download_video(youtube_link, video_dir_path)
    else:
        print("Video exists in folder.")

    if len(os.listdir(frame_folder_path)) != int(frame_number):
        frames = extract_frames_opencv(
            video_path=video_dir_path, 
            max_frames=frame_number, 
            trim_start=cf.TRIM_START_FRAMES,
            trim_end=cf.TRIM_END_FRAMES, 
            output_dir=frame_folder_path
        )
    else:
        print("Frames exist in folder.")
    

    # Call the Pixtral service function
    image_files = sorted([f for f in os.listdir(frame_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    process_with_pixtral(image_files, frame_folder_path, pixtral_response_path)

    return True
