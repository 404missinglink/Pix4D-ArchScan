# interface/gradio_interface.py

import gradio as gr
import logging

from services.video_processor import VideoProcessor

logger = logging.getLogger("DroneFootageSurveyor.interface.gradio_interface")

def create_gradio_interface():
    """
    Defines and returns the Gradio Blocks interface.
    """
    video_processor = VideoProcessor()

    with gr.Blocks() as iface:
        gr.Markdown("# 🚁 Drone Footage Surveyor\n"
                    "Upload drone video footage, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in a larger chat interface. The frames and their summaries will be saved automatically.")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="📥 Upload Drone Footage")
                max_frames_input = gr.Number(label="🔢 Max Frames to Extract", value=10, precision=0, step=1, interactive=True)
                include_images_input = gr.Checkbox(label="🖼️ Include Frame Images in Summaries", value=False)
                submit_btn = gr.Button("▶️ Process Video")
            with gr.Column(scale=3):  # Increased scale for a larger chatbot
                chatbot = gr.Chatbot(label="💬 Live Summarization")

        # Hidden states to store chat history and frame summaries
        chat_history = gr.State(value=[])
        frame_summaries = gr.State(value=[])

        # Define the generator function for streaming summaries to the chatbot
        submit_btn.click(
            fn=video_processor.process_video,
            inputs=[video_input, max_frames_input, include_images_input, chat_history, frame_summaries],
            outputs=[chatbot, frame_summaries],
            show_progress=True
        )

    return iface
