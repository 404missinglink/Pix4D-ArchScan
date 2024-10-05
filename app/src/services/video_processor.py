# services/video_processor.py

import os
import json
import time
import uuid
import logging
import traceback
from threading import Lock
from PIL import Image

from mistralai import Mistral

from utils.image_utils import encode_image
from utils.video_utils import save_temp_video
from frame_extractor import extract_frames_opencv
from config import (
    API_KEY,
    TEXT_MODEL,
    VISION_MODEL,
    RATE_LIMIT_SECONDS,
    TRIM_START_FRAMES,
    TRIM_END_FRAMES,
    BASE_FRAMES_DIR
)

logger = logging.getLogger("DroneFootageSurveyor.services.video_processor")

class VideoProcessor:
    def __init__(self):
        self.client = Mistral(api_key=API_KEY)
        self.rate_limit_lock = Lock()
        self.last_request_time = 0  # Timestamp of the last API request

    def process_video(self, video_path, max_frames=10, include_images=False, chat_history=None, frame_summaries=None):
        """
        Processes the uploaded video, updates the Chatbot history, saves frames with summaries to disk,
        and generates an overall summary using the larger text model.

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

            tmp_path = save_temp_video(video_path)
        except Exception as e:
            logger.error(f"Failed to save temporary video file: {str(e)}")
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
                                 "text":''' Provide a detailed analysis of this frame based on the following key areas relevant to structural and construction surveying. The response should be in JSON format with each key area as a separate field. Ensure the information is concise, direct, and easily processed by a larger language model. Additionally, identify and describe the type of project (e.g., building, bridge, road) present in the frame to provide more context.

                                            The JSON response should follow this structure:

                                            {
                                                "General Structural Condition": {
                                                    "Foundation": "<Description of the foundation condition>",
                                                    "Walls": "<Description of the walls condition>",
                                                    "Roof": "<Description of the roof condition>"
                                                },
                                                "External Features": {
                                                    "Façade & Cladding": "<Description of the façade and cladding>",
                                                    "Windows and Doors": "<Description of windows and doors condition>",
                                                    "Drainage and Gutters": "<Description of drainage and gutters>"
                                                },
                                                "Internal Condition": {
                                                    "Floors and Ceilings": "<Description of floors and ceilings>",
                                                    "Walls": "<Description of internal walls condition>",
                                                    "Electrical and Plumbing": "<Description of electrical and plumbing condition>"
                                                },
                                                "Signs of Water Damage or Moisture": {
                                                    "Stains or Discoloration": "<Description of water damage or discoloration>",
                                                    "Basement & Foundation": "<Description of any signs in the basement or foundation>"
                                                },
                                                "HVAC Systems": "<Description of HVAC systems if visible>",
                                                "Safety Features": {
                                                    "Fire Exits": "<Description of fire exits>",
                                                    "Handrails and Guardrails": "<Description of handrails and guardrails>"
                                                },
                                                "Landscaping & Surroundings": {
                                                    "Site Drainage": "<Description of site drainage>",
                                                    "Paths and Roads": "<Description of paths and roads>",
                                                    "Tree Proximity": "<Description of tree proximity>"
                                                },
                                                "Construction Progress (if an active project)": {
                                                    "Consistency with Plans": "<Assessment of consistency with plans>",
                                                    "Material Usage": "<Assessment of material usage>",
                                                    "Workmanship": "<Assessment of workmanship>"
                                                },
                                                "Temporary Supports & Site Safety (if under construction)": {
                                                    "Scaffolding": "<Description of scaffolding>",
                                                    "Temporary Structures": "<Description of temporary structures>"
                                                },
                                                "Building Services (if visible)": {
                                                    "Mechanical & Electrical Installations": "<Description of mechanical and electrical installations>",
                                                    "Elevators & Staircases": "<Description of elevators and staircases>"
                                                },
                                                "Project Type": "<Type of project identified, e.g., building, bridge, road>"
                                            '''},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ]

                # Implement consistent rate limiting: Ensure at least RATE_LIMIT_SECONDS between API requests
                with self.rate_limit_lock:
                    current_time = time.time()
                    elapsed_time = current_time - self.last_request_time
                    if elapsed_time < RATE_LIMIT_SECONDS:
                        sleep_time = RATE_LIMIT_SECONDS - elapsed_time
                        logger.info(f"Sleeping for {sleep_time:.2f} seconds to maintain rate limiting.")
                        time.sleep(sleep_time)
                    # Update the last_request_time to the current time before making the call
                    self.last_request_time = time.time()

                # Measure latency
                start_time = time.time()

                # Get the chat response with JSON format
                chat_response = self.client.chat.complete(
                    model=VISION_MODEL,
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
                        "Type of Project"  # Standardized key
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

        # After processing all frames, generate an overall summary and save it
        try:
            # Inform the user that an overall summary is being generated
            chat_history.append(("System", "📝 Generating an overall summary based on the analysis of all frames..."))
            yield [chat_history, frame_summaries]

            # Create a context by joining all frame summaries
            aggregated_summaries = "\n\n".join([frame['summary'] for frame in frames_data if 'summary' in frame])

            # Define the prompt for the text model
            prompt = (
                "From the following structural and construction survey summaries, provide a brief and actionable summary of key issues and recommended actions. "
                "Avoid unnecessary details and focus on practical recommendations. Follow this example structure:"
                "\n\n"
                "Example Structure:"
                "\n\n"
                "Key Issue Category (e.g., Foundation & Walls):"
                "\n- Issue: Brief description of the problem."
                "\n- Action: Clear, actionable recommendation to address the issue."
                "\n\n"
                "Key Issue Category (e.g., Roof):"
                "\n- Issue: Brief description of the problem."
                "\n- Action: Clear, actionable recommendation to address the issue."
                "\n\n"
                "Use this structure for all identified issues. Be concise and focus on critical actions."
                f"{aggregated_summaries}"
            )

            # Define messages for the chat completion
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in structural and construction surveying. You provide concise, practical recommendations without redundant or repeated information."},
                {"role": "user", "content": prompt}
            ]


            # Implement consistent rate limiting: Ensure at least RATE_LIMIT_SECONDS between API requests
            with self.rate_limit_lock:
                current_time = time.time()
                elapsed_time = current_time - self.last_request_time
                if elapsed_time < RATE_LIMIT_SECONDS:
                    sleep_time = RATE_LIMIT_SECONDS - elapsed_time
                    logger.info(f"Sleeping for {sleep_time:.2f} seconds to maintain rate limiting.")
                    time.sleep(sleep_time)
                # Update the last_request_time to the current time before making the call
                self.last_request_time = time.time()

            # Measure latency
            start_time = time.time()

            # Get the response from the text model
            overall_response = self.client.chat.complete(
                model=TEXT_MODEL,
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

            # Append the overall summary to frames_data for JSON
            frames_data_with_overall = {
                "run_id": run_id,
                "frames": frames_data,
                "overall_summary": overall_summary
            }

            # Save the frames_summary.json with overall summary
            json_path = os.path.join(run_dir, "frames_summary.json")
            with open(json_path, "w") as json_file:
                json.dump(frames_data_with_overall, json_file, indent=2)
            logger.info(f"Saved frames summary JSON with overall summary at {json_path}")

            # Yield the updated chat_history and frame_summaries
            yield [chat_history, frame_summaries]

        except Exception as e:
            logger.error(f"Error generating overall summary: {str(e)}")
            logger.debug(traceback.format_exc())
            chat_history.append(("System", "❌ An error occurred while generating the overall summary and potential solutions."))
            yield [chat_history, frame_summaries]
