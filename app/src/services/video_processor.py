# services/video_processor.py

import os
import json
import time
import uuid
import logging
import traceback
from threading import Lock
from PIL import Image
import requests  # Added for making HTTP requests

from mistralai import Mistral  # Retained for TEXT_MODEL

from utils.image_utils import encode_image
from utils.video_utils import save_temp_video
from frame_extractor import extract_frames_opencv
from config import (
    API_KEY,
    TEXT_MODEL,
    PIXTRAL_API_URL,  # New config for Pixtral's API endpoint
    RATE_LIMIT_SECONDS,
    TRIM_START_FRAMES,
    TRIM_END_FRAMES,
    BASE_FRAMES_DIR
)

logger = logging.getLogger("DroneFootageSurveyor.services.video_processor")

class VideoProcessor:
    def __init__(self):
        self.client = Mistral(api_key=API_KEY)  # Used for TEXT_MODEL
        self.rate_limit_lock = Lock()
        self.last_request_time = 0  # Timestamp of the last API request

    def process_video(self, video_path, max_frames=10, include_images=False, chat_history=None, frame_summaries=None):
        """
        Processes the uploaded video, updates the Chatbot history, saves frames with summaries to disk,
        and generates an overall summary using the larger text model.

        Yields:
            list: Updated chat history and updated frame summaries.
        """
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

                # Save the frame image to the run directory
                frame_filename = f"frame_{idx+1:03d}.jpg"
                frame_path = os.path.join(run_dir, frame_filename)
                img.save(frame_path, format='JPEG')
                logger.info(f"Saved frame {idx+1} as {frame_path}")

                # Construct the image URL using the base64 string
                image_url = f"data:image/jpeg;base64,{base64_image}"

                # Define the prompt for Pixtral
                prompt = (
                    '''
Provide a detailed analysis of this frame based on the following key areas relevant to structural and construction surveying. The response should be in plain text format with each key area as a separate section. Ensure the information is concise, direct, and easily processed by a larger language model. Additionally, identify and describe the type of project (e.g., building, bridge, road) present in the frame to provide more context.

### Key Areas:

- **General Structural Condition**: 
  - Foundation
  - Walls
  - Roof

- **External Features**: 
  - Façade & Cladding
  - Windows and Doors
  - Drainage and Gutters

- **Internal Condition**: 
  - Floors and Ceilings
  - Walls
  - Electrical and Plumbing

- **Signs of Water Damage or Moisture**: 
  - Stains or Discoloration
  - Basement & Foundation

- **HVAC Systems** (if visible)

- **Safety Features**: 
  - Fire Exits
  - Handrails and Guardrails

- **Landscaping & Surroundings**: 
  - Site Drainage
  - Paths and Roads
  - Tree Proximity

- **Construction Progress** (if an active project): 
  - Consistency with Plans
  - Material Usage
  - Workmanship

- **Temporary Supports & Site Safety** (if under construction): 
  - Scaffolding
  - Temporary Structures

- **Building Services** (if visible): 
  - Mechanical & Electrical Installations
  - Elevators & Staircases

- **Project Type**: Identify and describe the type of project (e.g., building, bridge, road).
'''
                )

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

                # Make the POST request to Pixtral's API
                response = requests.post(
                    PIXTRAL_API_URL,  # e.g., "http://127.0.0.1:5000/describe_image"
                    headers={"Content-Type": "application/json"},
                    json={
                        "image_url": image_url,
                        "prompt": prompt
                    }
                )

                end_time = time.time()
                latency = end_time - start_time  # in seconds

                # Log latency and image details
                height, width, channels = frame.shape
                logger.info(f"Frame {idx+1}: Width={width}px, Height={height}px, Channels={channels}, Latency={latency:.2f}s")

                if response.status_code != 200:
                    raise ValueError(f"Pixtral API returned status code {response.status_code}: {response.text}")

                chat_response = response.json()

                # Assuming Pixtral returns {"description": "<Plain text string>"}
                summary = chat_response.get("description", "")

                if not summary.strip():
                    raise ValueError("Pixtral API returned an empty summary.")

                # Append frame data to frames_data list
                frames_data.append({
                    "frame_number": idx + 1,
                    "frame_path": frame_path,
                    "summary": summary.strip()  # Store the plain text summary
                })

                # Format the summary for display
                if include_images and base64_image:
                    bot_message = (
                        f"### Frame {idx+1}\n\n"
                        f"![Frame {idx+1}](data:image/jpeg;base64,{base64_image})\n\n"
                        f"**Summary:**\n{summary.strip()}"
                    )
                else:
                    bot_message = f"### Frame {idx+1}\n\n**Summary:**\n{summary.strip()}"

                # Append the new summary to the history and frame_summaries
                chat_history.append(("System", bot_message))
                frame_summaries.append(summary.strip())  # Store the plain text summaries

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

            # Create a context by aggregating all frame summaries as plain text
            aggregated_summaries = "\n\n".join(frame_summaries)

            # Define the prompt for the text model
            prompt = (
                "From the following structural and construction survey summaries, generate a concise, yet detailed summary that includes both positive points and key issues with actionable recommendations. "
                "Ensure all points are relevant to the project type and provide value by focusing on critical improvements and benefits. Avoid repetition, and ensure the report reads smoothly while covering all important aspects. Do not contradict yourself. Use only information from the summaries below.\n\n"
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
