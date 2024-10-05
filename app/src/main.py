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

# Import the frame extraction function from the frame_extractor module
from frame_extractor import extract_frames_opencv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
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

def encode_image(image):
    """
    Encode the image to base64.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            with open(tmp.name, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(tmp.name)
            logger.debug("Image encoded to base64 successfully.")
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def process_video(video_path, max_frames=10, frame_interval=None):
    """
    Processes the uploaded video and returns a JSON summary.

    Parameters:
        video_path (str): Path to the uploaded video.
        max_frames (int): Maximum number of frames to extract.
        frame_interval (int, optional): Interval between frames to extract.

    Returns:
        dict: JSON summary of the video.
    """
    logger.info("Processing uploaded video.")
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            with open(video_path, "rb") as f:
                tmp.write(f.read())
            tmp_path = tmp.name
            logger.info(f"Temporary video saved at {tmp_path}")
    except Exception as e:
        logger.error(f"Failed to save temporary video file: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": "Failed to save the uploaded video."}

    try:
        frames = extract_frames_opencv(tmp_path, max_frames=max_frames, frame_interval=frame_interval)
    except Exception as e:
        logger.error(f"Failed to extract frames: {str(e)}")
        return {"error": "Failed to extract frames from the video."}
    finally:
        try:
            os.remove(tmp_path)
            logger.info(f"Temporary video file {tmp_path} removed.")
        except Exception as e:
            logger.warning(f"Could not remove temporary video file: {str(e)}")

    frame_summaries = []
    for idx, frame in enumerate(frames):
        logger.info(f"Processing frame {idx}")
        try:
            img = Image.fromarray(frame)
            base64_image = encode_image(img)
            if not base64_image:
                frame_summary = {"frame": idx, "error": "Failed to encode image."}
                frame_summaries.append(frame_summary)
                continue

            # Define the messages for the chat
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a summary of this frame."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Get the chat response with JSON format
            chat_response = client.chat.complete(
                model=vision_model,
                messages=messages,
                response_format={
                    "type": "json_object",
                }
            )
            summary = chat_response.choices[0].message.content

            # Validate if the response is valid JSON
            try:
                summary_json = json.loads(summary)
                frame_summaries.append({"frame": idx, "summary": summary_json})
                logger.info(f"Frame {idx} summarized successfully.")
            except json.JSONDecodeError:
                # If not valid JSON, include as plain text
                frame_summaries.append({"frame": idx, "summary": summary})
                logger.warning(f"Frame {idx} summary is not valid JSON.")

        except Exception as e:
            logger.error(f"Error processing frame {idx}: {str(e)}")
            logger.debug(traceback.format_exc())
            frame_summaries.append({"frame": idx, "error": str(e)})

    final_summary = {
        "video_summary": frame_summaries
    }
    logger.info("Video processing completed.")
    return final_summary

# Define Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Drone Footage"),
        gr.Number(label="Max Frames to Extract", value=10, precision=0, step=1),
        gr.Number(label="Frame Interval (Optional)", value=None, precision=0, step=1)
    ],
    outputs=gr.JSON(label="Live Summarization"),
    title="Drone Footage Summarizer",
    description="Upload drone video footage, specify the number of frames to extract and the interval, and Pixtral will provide live summarizations of the content in JSON format."
)

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio interface.")
        iface.launch(share=False)  # Set share=True to create a public link
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        logger.debug(traceback.format_exc())
