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
import uuid  # For unique run identifiers

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

# Base directory to save frames and summaries
BASE_FRAMES_DIR = "frames"

# Ensure the base frames directory exists
os.makedirs(BASE_FRAMES_DIR, exist_ok=True)

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

def process_video(video_path, max_frames=10, include_images=False, chat_history=None, frame_summaries=None):
    """
    Processes the uploaded video, updates the Chatbot history, and saves frames with summaries to disk.

    Parameters:
        video_path (str): Path to the uploaded video.
        max_frames (int): Maximum number of frames to extract.
        include_images (bool): Whether to include images in the summaries.
        chat_history (list): Accumulated chat history as a list of (user, bot) tuples.
        frame_summaries (list): Accumulated frame summaries.

    Yields:
        list: Updated chat history and updated frame summaries.
    """
    global last_request_time  # Access the global variable for rate limiting
    logger.info("Processing uploaded video.")
    frames = []  # Initialize frames to ensure it's always defined

    # Initialize chat_history and frame_summaries if None
    if chat_history is None:
        chat_history = []
    if frame_summaries is None:
        frame_summaries = []

    # Generate a unique identifier for this run
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(BASE_FRAMES_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Created run directory at {run_dir}")

    # Initialize a list to store frame data for JSON
    frames_data = []

    try:
        # Inform the user that processing has started
        chat_history.append(("System", "📹 Processing your video. Please wait..."))
        yield [chat_history, frame_summaries]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            with open(video_path, "rb") as f:
                tmp.write(f.read())
            tmp_path = tmp.name
            logger.info(f"Temporary video saved at {tmp_path}")
    except Exception as e:
        logger.error(f"Failed to save temporary video file: {str(e)}")
        logger.debug(traceback.format_exc())
        chat_history.append(("System", "❌ Failed to save the uploaded video."))
        yield [chat_history, frame_summaries]
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
        logger.info(f"Extracted {len(frames)} frames from the video.")
    except Exception as e:
        logger.error(f"Failed to extract frames: {str(e)}")
        chat_history.append(("System", "❌ Failed to extract frames from the video."))
        yield [chat_history, frame_summaries]
    finally:
        try:
            os.remove(tmp_path)
            logger.info(f"Temporary video file {tmp_path} removed.")
        except Exception as e:
            logger.warning(f"Could not remove temporary video file: {str(e)}")

    if not frames:
        chat_history.append(("System", "⚠️ No frames were extracted from the video."))
        yield [chat_history, frame_summaries]
        return

    for idx, frame in enumerate(frames):
        logger.info(f"Processing frame {idx+1}/{len(frames)}")
        try:
            img = Image.fromarray(frame)
            base64_image = encode_image(img)  # Always encode the image

            if base64_image is None:
                raise ValueError("Image encoding failed.")

            # Define the messages for the chat with enhanced JSON instruction
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Provide a detailed analysis of this frame based on the following key areas relevant to structural and construction surveying. "
                                "The response should be in JSON format with each key area as a separate field. "
                                "Ensure the information is concise, direct, and easily processed by a larger language model.\n\n"
                                "1. General Structural Condition\n"
                                "   - Foundation\n"
                                "   - Walls\n"
                                "   - Roof\n"
                                "2. External Features\n"
                                "   - Façade & Cladding\n"
                                "   - Windows and Doors\n"
                                "   - Drainage and Gutters\n"
                                "3. Internal Condition\n"
                                "   - Floors and Ceilings\n"
                                "   - Walls\n"
                                "   - Electrical and Plumbing\n"
                                "4. Signs of Water Damage or Moisture\n"
                                "   - Stains or Discoloration\n"
                                "   - Basement & Foundation\n"
                                "5. HVAC Systems\n"
                                "6. Safety Features\n"
                                "   - Fire Exits\n"
                                "   - Handrails and Guardrails\n"
                                "7. Landscaping & Surroundings\n"
                                "   - Site Drainage\n"
                                "   - Paths and Roads\n"
                                "   - Tree Proximity\n"
                                "8. Construction Progress (if an active project)\n"
                                "   - Consistency with Plans\n"
                                "   - Material Usage\n"
                                "   - Workmanship\n"
                                "9. Temporary Supports & Site Safety (if under construction)\n"
                                "   - Scaffolding\n"
                                "   - Temporary Structures\n"
                                "10. Building Services (if visible)\n"
                                "    - Mechanical & Electrical Installations\n"
                                "    - Elevators & Staircases\n\n"
                                "Additionally, identify and describe the type of project (e.g., building, bridge, road) present in the frame to provide more context."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Implement consistent rate limiting: Ensure at least 1 second between API requests
            with rate_limit_lock:
                current_time = time.time()
                elapsed_time = current_time - last_request_time
                if elapsed_time < 1.0:
                    sleep_time = 1.0 - elapsed_time
                    logger.info(f"Sleeping for {sleep_time:.2f} seconds to maintain rate limiting.")
                    time.sleep(sleep_time)
                # Update the last_request_time to the current time before making the call
                last_request_time = time.time()

            # Measure latency
            start_time = time.time()

            # Get the chat response with JSON format
            chat_response = client.chat.complete(
                model=vision_model,
                messages=messages,
                response_format={
                    "type": "json_object",
                },
                temperature=0  # Set temperature to zero for determinism
            )

            end_time = time.time()
            latency = end_time - start_time  # in seconds

            # Log latency and image details
            height, width, channels = frame.shape
            logger.info(f"Frame {idx+1}: Width={width}px, Height={height}px, Channels={channels}, Latency={latency:.2f}s")

            summary = chat_response.choices[0].message.content

            # Validate if the response is valid JSON and a dictionary
            try:
                summary_json = json.loads(summary)
                if not isinstance(summary_json, dict):
                    raise ValueError("Parsed JSON is not a dictionary.")

                # Ensure all key areas are present in the JSON
                required_keys = [
                    "General Structural Condition",
                    "External Features",
                    "Internal Condition",
                    "Signs of Water Damage or Moisture",
                    "HVAC Systems",
                    "Safety Features",
                    "Landscaping & Surroundings",
                    "Construction Progress",
                    "Temporary Supports & Site Safety",
                    "Building Services",
                    "Project Type"  # New key added for project type
                ]
                for key in required_keys:
                    if key not in summary_json:
                        summary_json[key] = "N/A"
                summary_text = json.dumps(summary_json, indent=2)
                logger.info(f"Frame {idx+1} summarized successfully.")
            except (json.JSONDecodeError, ValueError) as e:
                # If not valid JSON or not a dict, include as plain text with error
                summary_text = "❌ Invalid JSON format received."
                raw_response = summary
                logger.warning(f"Frame {idx+1} summary is not valid JSON or not a dictionary.")
                chat_history.append(("System", f"❌ Frame {idx+1}: Received summary is not valid JSON or not a dictionary.\n\n**Raw Response:**\n```json\n{raw_response}\n```"))
                yield [chat_history, frame_summaries]
                continue

            # Save the frame image to the run directory
            frame_filename = f"frame_{idx+1:03d}.jpg"
            frame_path = os.path.join(run_dir, frame_filename)
            img.save(frame_path, format='JPEG')
            logger.info(f"Saved frame {idx+1} as {frame_path}")

            # Append frame data to frames_data list
            frames_data.append({
                "frame_number": idx + 1,
                "frame_path": frame_path,
                "summary": summary_text
            })

            # Format the summary for display
            if include_images and base64_image:
                bot_message = f"### Frame {idx+1}\n\n![Frame {idx+1}](data:image/jpeg;base64,{base64_image})\n\n**Summary:**\n```json\n{summary_text}\n```"
            else:
                bot_message = f"### Frame {idx+1}\n\n**Summary:**\n```json\n{summary_text}\n```"

            # Append the new summary to the history and frame_summaries
            chat_history.append(("System", bot_message))
            frame_summaries.append(summary_text)  # Store only the text summaries

            # Yield the updated chat_history and frame_summaries
            yield [chat_history, frame_summaries]

        except Exception as e:
            logger.error(f"Error processing frame {idx+1}: {str(e)}")
            logger.debug(traceback.format_exc())
            # Determine specific error types for better feedback
            if isinstance(e, TypeError):
                frame_error_md = f"❌ Frame {idx+1}: Type error encountered. Details: {str(e)}"
            elif isinstance(e, ValueError):
                frame_error_md = f"❌ Frame {idx+1}: Value error encountered. Details: {str(e)}"
            else:
                frame_error_md = f"❌ Frame {idx+1}: {str(e)}"
            chat_history.append(("System", frame_error_md))
            yield [chat_history, frame_summaries]

    # After processing all frames, save the JSON file with frame paths and summaries
    try:
        json_data = {
            "run_id": run_id,
            "frames": frames_data
        }
        json_path = os.path.join(run_dir, "frames_summary.json")
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=2)
        logger.info(f"Saved frames summary JSON at {json_path}")
    except Exception as e:
        logger.error(f"Error saving frames summary JSON: {str(e)}")
        logger.debug(traceback.format_exc())
        chat_history.append(("System", "❌ An error occurred while saving frames summaries to disk."))
        yield [chat_history, frame_summaries]
        return

    # Finally, inform the user that processing is completed without an additional message
    # (As per previous instructions)
    yield [chat_history, frame_summaries]

def handle_user_question(user_input, chat_history, frame_summaries):
    """
    This function is no longer needed as we're removing the Q&A feature.
    """
    pass  # Or remove this function entirely

# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# 🚁 Drone Footage Surveyor\nUpload drone video footage, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in a larger chat interface. The frames and their summaries will be saved automatically.")

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

    # Removed the "Ask a Question" section
    # with gr.Row():
    #     with gr.Column(scale=3):
    #         user_question = gr.Textbox(
    #             label="❓ Ask a Question",
    #             placeholder="Type your question here...",
    #             interactive=True
    #         )
    #         send_btn = gr.Button("Send")

    # Define the generator function for streaming summaries to the chatbot
    submit_btn.click(
        fn=process_video,
        inputs=[video_input, max_frames_input, include_images_input, chat_history, frame_summaries],
        outputs=[chatbot, frame_summaries],
        show_progress=True
    )

    # Removed the user question handlers
    # send_btn.click(
    #     fn=handle_user_question,
    #     inputs=[user_question, chat_history, frame_summaries],
    #     outputs=[chatbot, frame_summaries],
    #     queue=True  # Enable queuing for better performance
    # )

    # Allow pressing Enter in the textbox to send the question
    # user_question.submit(
    #     fn=handle_user_question,
    #     inputs=[user_question, chat_history, frame_summaries],
    #     outputs=[chatbot, frame_summaries],
    #     queue=True
    # )

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio interface.")
        iface.launch(share=False)  # Set share=True to create a public link
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        logger.debug(traceback.format_exc())
