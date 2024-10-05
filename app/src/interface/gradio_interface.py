import gradio as gr
import logging
import os
from datetime import datetime
from services.video_processor import VideoProcessor
from config import UPLOAD_VIDEOS_FOLDER  # Importing from config.py
from utils.download_video import download_video  # Importing from download_video.py

logger = logging.getLogger("DroneFootageSurveyor.interface.gradio_interface")

def ensure_destination_folder_exists(folder_path):
    """Ensure that the destination folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def generate_timestamped_filename():
    """Generate a timestamped filename for the video."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"video_{timestamp}.mp4", timestamp

def get_video_id_from_youtube_link(youtube_link):
    """Extract the video ID from the YouTube link."""
    return youtube_link.split('=')[-1]

def process_input(input_type, youtube_link, local_video, max_frames, include_images, chat_history):
    """Process either YouTube link or local video, and return saved file path and chatbot messages."""
    
    video_processor = VideoProcessor()
    
    if input_type == "YouTube Link":
        # Extract the video ID from the YouTube link
        video_id = get_video_id_from_youtube_link(youtube_link)
        video_folder = os.path.join(UPLOAD_VIDEOS_FOLDER, video_id)
        
        # Ensure that the destination folder for this video exists
        ensure_destination_folder_exists(video_folder)
        
        # Check if the video has already been processed
        video_filename = f"{video_id}.mp4"
        output_path1 = os.path.join(video_folder, video_filename)
        output_path = os.path.join(output_path1, youtube_link.split('=')[-1]+'.mp4')
        
        if not os.path.exists(output_path):  # Only download if the video doesn't exist
            summary_message = f"Processing YouTube video: {youtube_link}"
            # Use the download_video function from download_video.py
            download_video(youtube_link, output_path)
        else:
            summary_message = f"Video {video_id} has already been processed. Skipping download."
    
    elif input_type == "Local File":
        # For local files, we'll save it with a timestamp in the main folder
        video_filename, timestamp = generate_timestamped_filename()
        summary_message = f"Processing local video file: {video_filename}"
        
        # Process the local video file using the existing function
        video_processor.process_video(local_video, output_path, max_frames, include_images)
    
    # Add the result to the chat history
    chat_history.append(("System", summary_message))
    
    return chat_history, output_path

def create_gradio_interface():
    """
    Defines and returns the Gradio Blocks interface with enhanced layout and larger chatbot window.
    """
    video_processor = VideoProcessor()

    with gr.Blocks() as iface:
        # Improved header markdown with larger font and an appealing subheading
        gr.Markdown("""
        <h1 style='font-size: 2.5em; text-align: center;'>🚁 PIX4D-ARCHSCAN</h1>
        <p style='font-size: 1.2em; text-align: center;'>Upload drone video footage, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in a larger chat interface. The frames and their summaries will be saved automatically.</p>
        """, elem_id="header")

        # Adjusted layout to make the chatbot larger and other elements more visually balanced
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):  # Keeping the input section compact
                input_type = gr.Dropdown(label="Choose Input Type", choices=["YouTube Link", "Local File"], value="YouTube Link")
                youtube_link = gr.Textbox(label="📥 YouTube Video Link", placeholder="Enter YouTube video URL here", visible=True)
                local_video = gr.Video(label="📁 Upload Local Video File", visible=False)
                max_frames_input = gr.Number(label="🔢 Max Frames to Extract", value=10, precision=0, step=1, interactive=True)
                include_images_input = gr.Checkbox(label="🖼️ Include Frame Images in Summaries", value=False)
                submit_btn = gr.Button("▶️ Process Video", elem_classes="submit-btn")
                
            # Expanded scale to make the chatbot area larger
            with gr.Column(scale=5, min_width=600):
                chatbot = gr.Chatbot(label="💬 Live Summarization", elem_id="chatbot", height=500)  # Larger height for better content display

        # Hidden states to store chat history and frame summaries
        chat_history = gr.State(value=[])
        frame_summaries = gr.State(value=[])

        # Control visibility of input fields based on selection
        def toggle_input_fields(selected_input_type):
            if selected_input_type == "YouTube Link":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        # Update the visibility of YouTube link and local video inputs
        input_type.change(toggle_input_fields, inputs=input_type, outputs=[youtube_link, local_video])

        # Define the generator function for streaming summaries to the chatbot
        submit_btn.click(
            fn=process_input,
            inputs=[input_type, youtube_link, local_video, max_frames_input, include_images_input, chat_history],
            outputs=[chatbot, frame_summaries],
            show_progress=True
        )

    return iface
