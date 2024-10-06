from services.main_service import main_service
import config.config as cf
from time import time
import os
import gradio as gr
from services.interface_service import create_gradio_interface  # Import the Gradio interface

def process_input(youtube_link, frame_number, include_images, chat_history):
    """
    Process the YouTube video link, extract frames, and return summarizations.
    """
    video_id = youtube_link.split('=')[-1]
    dir_path = os.path.join("data", str(video_id))
    os.makedirs(dir_path, exist_ok=True)

    video_dir_path = os.path.join(cf.video_folder_path, f"{video_id}.mp4")

    # Call the main service to process the video
    main_service(youtube_link, video_dir_path, frame_number, dir_path)

    # For now, let's simulate a basic response (replace this with actual summarization)
    response = f"Processed video {video_id} with {frame_number} frames. Include images: {include_images}."

    # Append response to chat history
    chat_history.append((f"User: Process {frame_number} frames", f"System: {response}"))

    return chat_history, []  # Returning empty frame_summaries for now

if __name__ == "__main__":
    # Pass process_input to the Gradio interface
    iface = create_gradio_interface(process_input)
    iface.launch()
