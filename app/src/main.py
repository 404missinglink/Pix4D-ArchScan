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

def process_video(video_path, max_frames=10, include_images=False, chat_history=None, frame_summaries=None):
    """
    Processes the uploaded video and updates the Chatbot history and frame summaries in real-time.

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

    # After processing all frames, generate an overall summary and potential solutions
    try:
        # Inform the user that an overall summary is being generated
        chat_history.append(("System", "📝 Generating an overall summary and potential solutions based on the analysis of all frames..."))
        yield [chat_history, frame_summaries]

        # Create a context by joining all frame summaries
        aggregated_summaries = "\n\n".join(frame_summaries)

        # Define the prompt for the text model
        prompt = (
            "Based on the following frame summaries from a structural and construction survey, provide a comprehensive overall summary highlighting the key issues identified. "
            "Additionally, suggest potential solutions or actions to address these issues. Ensure the response is clear, concise, and actionable.\n\n"
            f"{aggregated_summaries}"
        )

        # Define messages for the chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in structural and construction surveying."},
            {"role": "user", "content": prompt}
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

        # Get the response from the text model
        overall_response = client.chat.complete(
            model=text_model,
            messages=messages,
            temperature=0  # Set temperature to zero for determinism
        )

        end_time = time.time()
        latency = end_time - start_time  # in seconds

        logger.info(f"Overall Summary Latency: {latency:.2f}s")

        overall_summary = overall_response.choices[0].message.content.strip()

        # Format the overall summary for display
        overall_summary_md = f"### Overall Summary and Potential Solutions\n\n{overall_summary}"

        # Append the overall summary to the chat_history
        chat_history.append(("System", overall_summary_md))

        # Yield the updated chat_history and frame_summaries
        yield [chat_history, frame_summaries]

    except Exception as e:
        logger.error(f"Error generating overall summary: {str(e)}")
        logger.debug(traceback.format_exc())
        chat_history.append(("System", "❌ An error occurred while generating the overall summary and potential solutions."))
        yield [chat_history, frame_summaries]

    # Finally, inform the user that processing is completed
    completion_md = "✅ **Processing Completed**\n\nAll frame summaries and the overall analysis have been generated."
    chat_history.append(("System", completion_md))
    yield [chat_history, frame_summaries]

def handle_user_question(user_input, chat_history, frame_summaries):
    """
    Handles user questions by generating responses based on frame summaries.

    Parameters:
        user_input (str): The user's question.
        chat_history (list): The current chat history.
        frame_summaries (list): The list of frame summaries.

    Returns:
        list: Updated chat history and unchanged frame summaries.
    """
    if not user_input.strip():
        return [chat_history, frame_summaries]  # Ignore empty messages

    # Append user message to chat_history
    chat_history.append((user_input, None))

    # Create a context by joining all frame summaries
    context = "\n\n".join([f"Frame {i+1}: {summary}" for i, summary in enumerate(frame_summaries)])

    # Define the prompt for the text model
    prompt = (
        "You are an assistant that answers questions based on the following frame summaries:\n\n"
        f"{context}\n\n"
        "User Question: {user_question}\n\n"
        "Answer:"
    ).format(user_question=user_input)

    # Define messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in structural and construction surveying."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Implement consistent rate limiting: Ensure at least 1 second between API requests
        global last_request_time
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

        # Get the response from the text model
        response = client.chat.complete(
            model=text_model,
            messages=messages,
            temperature=0  # Set temperature to zero for determinism
        )

        end_time = time.time()
        latency = end_time - start_time  # in seconds

        logger.info(f"User Question Latency: {latency:.2f}s")

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error handling user question: {str(e)}")
        logger.debug(traceback.format_exc())
        answer = "❌ An error occurred while processing your question."

    # Append bot response to chat_history
    chat_history[-1] = (user_input, answer)

    # Yield the updated chat_history and unchanged frame_summaries
    return [chat_history, frame_summaries]

# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# 🚁 Drone Footage Surveyor\nUpload drone video footage, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in a chat interface. After processing, you can ask questions about the summaries.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📥 Upload Drone Footage")
            max_frames_input = gr.Number(label="🔢 Max Frames to Extract", value=10, precision=0, step=1, interactive=True)
            include_images_input = gr.Checkbox(label="🖼️ Include Frame Images in Summaries", value=False)
            submit_btn = gr.Button("▶️ Process Video")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="💬 Live Summarization")

    # Hidden states to store chat history and frame summaries
    chat_history = gr.State(value=[])
    frame_summaries = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=3):
            user_question = gr.Textbox(
                label="❓ Ask a Question",
                placeholder="Type your question here...",
                interactive=True
            )
            send_btn = gr.Button("Send")

    # Define the generator function for streaming summaries to the chatbot
    submit_btn.click(
        fn=process_video,
        inputs=[video_input, max_frames_input, include_images_input, chat_history, frame_summaries],
        outputs=[chatbot, frame_summaries],
        show_progress=True
    )

    # Define the function to handle user questions
    send_btn.click(
        fn=handle_user_question,
        inputs=[user_question, chat_history, frame_summaries],
        outputs=[chatbot, frame_summaries],
        queue=True  # Enable queuing for better performance
    )

    # Allow pressing Enter in the textbox to send the question
    user_question.submit(
        fn=handle_user_question,
        inputs=[user_question, chat_history, frame_summaries],
        outputs=[chatbot, frame_summaries],
        queue=True
    )

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio interface.")
        iface.launch(share=False)  # Set share=True to create a public link
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        logger.debug(traceback.format_exc())
