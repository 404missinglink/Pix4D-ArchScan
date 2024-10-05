import os
import gradio as gr
from mistralai import Mistral
from PIL import Image
import tempfile
import base64
from dotenv import load_dotenv
import cv2
import logging
import traceback
import json

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

def extract_frames_opencv(video_path, max_frames=10):
    """
    Extracts frames from the video at regular intervals using OpenCV.
    """
    logger.info(f"Starting frame extraction from {video_path} with max_frames={max_frames}")
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps else 0

        logger.info(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration}s")

        frame_interval = max(total_frames // max_frames, 1)

        frame_indices = [i * frame_interval for i in range(max_frames)]
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                logger.debug(f"Extracted frame {idx}")
            else:
                logger.warning(f"Failed to read frame at index {idx}")
        cap.release()
        logger.info(f"Extracted {len(frames)} frames.")
    except Exception as e:
        logger.error(f"Error during frame extraction: {str(e)}")
        logger.debug(traceback.format_exc())
        raise e
    return frames

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

def process_video(video_path):
    """
    Processes the uploaded video and returns a JSON summary.
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
        frames = extract_frames_opencv(tmp_path, max_frames=10)
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
    inputs=gr.Video(label="Upload Drone Footage"),
    outputs=gr.JSON(label="Live Summarization"),
    title="Drone Footage Summarizer",
    description="Upload drone video footage, and Pixtral will provide live summarizations of the content in JSON format."
)

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio interface.")
        iface.launch(share=False)  # Set share=True to create a public link
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        logger.debug(traceback.format_exc())
