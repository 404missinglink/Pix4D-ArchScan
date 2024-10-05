import gradio as gr
import logging

from services.video_processor import VideoProcessor

logger = logging.getLogger("DroneFootageSurveyor.interface.gradio_interface")

def create_gradio_interface():
    """
    Defines and returns the Gradio Blocks interface with enhanced layout and larger chatbot window.
    """
    video_processor = VideoProcessor()

    with gr.Blocks() as iface:
        # Improved header markdown with larger font and an appealing subheading
        gr.Markdown("""
        <h1 style='font-size: 2.5em; text-align: center;'>🚁 Drone Footage Surveyor</h1>
        <p style='font-size: 1.2em; text-align: center;'>Enter a YouTube video link, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in a larger chat interface. The frames and their summaries will be saved automatically.</p>
        """, elem_id="header")

        # Adjusted layout to make the chatbot larger and other elements more visually balanced
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):  # Keeping the input section compact
                video_input = gr.Textbox(label="📥 YouTube Video Link", placeholder="Enter YouTube video URL here")
                max_frames_input = gr.Number(label="🔢 Max Frames to Extract", value=10, precision=0, step=1, interactive=True)
                include_images_input = gr.Checkbox(label="🖼️ Include Frame Images in Summaries", value=False)
                submit_btn = gr.Button("▶️ Process Video", elem_classes="submit-btn")
                
            # Expanded scale to make the chatbot area larger
            with gr.Column(scale=5, min_width=600):
                chatbot = gr.Chatbot(label="💬 Live Summarization", elem_id="chatbot", height=500)  # Larger height for better content display

        # Hidden states to store chat history and frame summaries
        chat_history = gr.State(value=[])
        frame_summaries = gr.State(value=[])

        # Define the generator function for streaming summaries to the chatbot
        submit_btn.click(
            fn=video_processor.process_video,  # You need to modify this function to handle the YouTube link
            inputs=[video_input, max_frames_input, include_images_input, chat_history, frame_summaries],
            outputs=[chatbot, frame_summaries],
            show_progress=True
        )

    return iface
