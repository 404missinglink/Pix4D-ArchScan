# main_app.py

import os
import gradio as gr
from mistralai import Mistral
from PIL import Image
import tempfile
import base64
from dotenv import load_dotenv
import logging
import traceback
import json
import time  # For latency measurement and rate limiting
from threading import Lock  # For thread-safe rate limiting

# Import the frame extraction function from the frame_extractor module
from frame_extractor import extract_frames_opencv

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture general information
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to file
        logging.StreamHandler()          # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("API_KEY")
if not api_key:
    logger.error("API_KEY not found in environment variables.")
    raise ValueError("Please set the API_KEY environment variable in the .env file.")

# Specify models
text_model = "mistral-large-latest"
vision_model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Initialize a lock for rate limiting to ensure thread safety
rate_limit_lock = Lock()
last_request_time = 0  # Timestamp of the last API request

# Constants for trimming frames
TRIM_START_FRAMES = 30  # Number of frames to trim from the start
TRIM_END_FRAMES = 30    # Number of frames to trim from the end

def encode_image(image):
    """
    Encode the image to base64.
    
    Parameters:
        image (PIL.Image): The image to encode.
    
    Returns:
        str: Base64-encoded string of the image.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format='JPEG')
            with open(tmp.name, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(tmp.name)
            logger.debug("Image encoded to base64 successfully.")
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def process_video(video_path, max_frames=10, include_images=False, history=""):
    """
    Processes the uploaded video and updates the Markdown summary history in real-time.
    
    Parameters:
        video_path (str): Path to the uploaded video.
        max_frames (int): Maximum number of frames to extract.
        include_images (bool): Whether to include images in the summaries.
        history (str): Accumulated Markdown summaries.
    
    Yields:
        Tuple containing:
            - Updated Markdown summaries.
            - Updated history state.
    """
    global last_request_time  # Access the global variable for rate limiting
    logger.info("Processing uploaded video.")
    frames = []  # Initialize frames to ensure it's always defined
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            with open(video_path, "rb") as f:
                tmp.write(f.read())
            tmp_path = tmp.name
            logger.info(f"Temporary video saved at {tmp_path}")
    except Exception as e:
        logger.error(f"Failed to save temporary video file: {str(e)}")
        logger.debug(traceback.format_exc())
        error_md = "## Error\n\nFailed to save the uploaded video.\n"
        history += error_md
        yield history, history
        return

    try:
        # Extract frames with dynamic interval and trimming
        frames = extract_frames_opencv(
            video_path=tmp_path,
            max_frames=max_frames,
            frame_interval=None,  # Let frame_extractor handle it
            trim_start=TRIM_START_FRAMES,
            trim_end=TRIM_END_FRAMES
        )
    except Exception as e:
        logger.error(f"Failed to extract frames: {str(e)}")
        error_md = "## Error\n\nFailed to extract frames from the video.\n"
        history += error_md
        yield history, history
    finally:
        try:
            os.remove(tmp_path)
            logger.info(f"Temporary video file {tmp_path} removed.")
        except Exception as e:
            logger.warning(f"Could not remove temporary video file: {str(e)}")

    if not frames:
        no_frames_md = "## No Frames Extracted\n\nNo frames were extracted from the video.\n"
        history += no_frames_md
        yield history, history

    # Initialize history if it's empty
    if not history:
        history = "# Video Summary\n\nProcessing summaries...\n\n"

    for idx, frame in enumerate(frames):
        logger.info(f"Processing frame {idx}")
        try:
            img = Image.fromarray(frame)
            base64_image = encode_image(img)  # Always encode the image

            # Define the messages for the chat with strict JSON instruction
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a concise summary of this frame strictly in JSON format. The JSON should contain keys like 'description'. Focus on surveys of areas, buildings, and construction. Keep it short and informative."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Implement rate limiting: Ensure at least 1 second between API requests
            with rate_limit_lock:
                current_time = time.time()
                elapsed_time = current_time - last_request_time
                if elapsed_time < 1:
                    sleep_time = 1.4 - elapsed_time  # Corrected to 1 second
                    logger.info(f"Rate limiting in effect. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                # Update the last_request_time to the current time after sleeping
                last_request_time = time.time()

            # Measure latency
            start_time = time.time()

            # Get the chat response with JSON format
            chat_response = client.chat.complete(
                model=vision_model,
                messages=messages,
                response_format={
                    "type": "json_object",
                }
            )

            end_time = time.time()
            latency = end_time - start_time  # in seconds

            # Log latency and image details
            height, width, channels = frame.shape
            logger.info(f"Frame {idx}: Width={width}px, Height={height}px, Channels={channels}, Latency={latency:.2f}s")

            summary = chat_response.choices[0].message.content

            # Validate if the response is valid JSON
            try:
                summary_json = json.loads(summary)
                # Extract relevant information
                description = summary_json.get("description", "N/A")

                summary_text = f"**Description:** {description}"
                logger.info(f"Frame {idx} summarized successfully.")
            except json.JSONDecodeError:
                # If not valid JSON, include as plain text with error
                summary_text = "Invalid JSON format received."
                frame_summary_md = f"### Frame {idx}\n\n**Error**: Received summary is not valid JSON.\n\n**Raw Response:**\n\n{summary}\n\n"
                history += frame_summary_md
                logger.warning(f"Frame {idx} summary is not valid JSON.")
                yield history, history
                continue

            # Format the summary in Markdown
            if include_images and base64_image:
                frame_summary_md = f"### Frame {idx}\n\n![Frame {idx}](data:image/jpeg;base64,{base64_image})\n\n**Summary:**\n\n{summary_text}\n\n"
            else:
                frame_summary_md = f"### Frame {idx}\n\n**Summary:**\n\n{summary_text}\n\n"

            # Append the new summary to the history
            history += frame_summary_md

            # Yield the updated history
            yield history, history

        except Exception as e:
            logger.error(f"Error processing frame {idx}: {str(e)}")
            logger.debug(traceback.format_exc())
            frame_error_md = f"### Frame {idx}\n\n**Error**: {str(e)}\n\n"
            history += frame_error_md
            yield history, history

    completion_md = "\n\n# Processing Completed\n\nAll frame summaries have been generated."
    history += completion_md
    yield history, history

# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Drone Footage Summarizer\nUpload drone video footage, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in Markdown format.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Drone Footage")
            max_frames_input = gr.Number(label="Max Frames to Extract", value=10, precision=0, step=1)
            # Removed frame_interval_input as it's now handled by frame_extractor.py
            # include_images_input remains to control image display
            include_images_input = gr.Checkbox(label="Include Frame Images in Summaries", value=False)
            submit_btn = gr.Button("Process Video")
        with gr.Column():
            # Simple Markdown component for summaries
            summary_output = gr.Markdown(label="Live Summarization", value="")

    # Hidden state to store history
    summary_history = gr.State(value="")

    # Define the generator function for streaming summaries
    submit_btn.click(
        fn=process_video,
        inputs=[video_input, max_frames_input, include_images_input, summary_history],
        outputs=[summary_output, summary_history],
        show_progress=True
    )

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio interface.")
        iface.launch(share=False)  # Set share=True to create a public link
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        logger.debug(traceback.format_exc())
