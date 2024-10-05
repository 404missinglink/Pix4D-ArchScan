# frame_extractor.py

import cv2
import logging
import traceback
import os

# Setup logging for the frame extractor module
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture general information
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)
logger = logging.getLogger(__name__)

def extract_frames_opencv(video_path, max_frames=10, frame_interval=None, trim_start=30, trim_end=30, output_dir=None):
    """
    Extracts frames from a video by dynamically adjusting the frame interval based on trimmed frames
    and saves them into a specified output directory.
    
    Parameters:
        video_path (str): Path to the input video file.
        max_frames (int): Maximum number of frames to extract from the video.
        frame_interval (int, optional): Specific interval between frames to extract.
                                        If None, it is automatically calculated based on max_frames.
        trim_start (int): Number of frames to trim from the beginning of the video (to exclude intro).
        trim_end (int): Number of frames to trim from the end of the video (to exclude outro).
        output_dir (str, optional): Directory where extracted frames will be saved.
                                    If None, frames are not saved to disk.
    
    Returns:
        List of frames in RGB format. Each frame is a NumPy array.
    """
    logger.info(f"Starting frame extraction from '{video_path}' with max_frames={max_frames}, frame_interval={frame_interval}, trim_start={trim_start}, trim_end={trim_end}, output_dir={output_dir}")
    frames = []  # List to store the extracted frames

    try:
        # Initialize the video capture object
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        # Retrieve video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
        duration = total_frames / fps if fps else 0  # Duration of the video in seconds

        logger.info(f"Video Properties - FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

        # Ensure trim_start and trim_end do not exceed total_frames
        trim_start = min(trim_start, total_frames // 2)
        trim_end = min(trim_end, total_frames - trim_start)

        logger.info(f"Trimming {trim_start} frames from the start and {trim_end} frames from the end.")

        # Calculate remaining frames after trimming
        remaining_frames = total_frames - trim_start - trim_end
        if remaining_frames <= 0:
            logger.warning("After trimming, no frames remain to extract.")
            return frames  # Return empty list

        logger.info(f"Remaining frames after trimming: {remaining_frames}")

        # Calculate frame_interval if not provided
        if frame_interval is None:
            frame_interval = max(remaining_frames // max_frames, 1)  # Ensure at least interval of 1
            logger.info(f"Calculated frame_interval based on max_frames: {frame_interval}")
        else:
            logger.info(f"Using provided frame_interval: {frame_interval}")

        # Adjust max_frames if frame_interval is too large
        possible_max_frames = (remaining_frames + frame_interval - 1) // frame_interval  # Ceiling division
        if possible_max_frames < max_frames:
            logger.warning(f"Requested max_frames={max_frames} is greater than possible_max_frames={possible_max_frames} with frame_interval={frame_interval}. Adjusting max_frames to {possible_max_frames}.")
            max_frames = possible_max_frames

        current_frame = trim_start  # Start after trimming

        # Ensure output_dir exists if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Frames will be saved to '{output_dir}'.")

        # Loop through the video and extract frames at specified intervals
        while current_frame < total_frames - trim_end and len(frames) < max_frames:
            # Set the position of the next frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()  # Read the frame
            if ret:
                # Convert the frame from BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Log the size of the extracted frame
                height, width, channels = frame_rgb.shape
                logger.info(f"Extracted Frame {current_frame}: Width={width}px, Height={height}px, Channels={channels}")

                frames.append(frame_rgb)  # Add the frame to the list
                logger.debug(f"Frame {current_frame} added to the frames list.")

                # Save frame as image if output_dir is specified
                if output_dir:
                    frame_filename = os.path.join(output_dir, f"frame_{current_frame}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    logger.debug(f"Frame {current_frame} saved as '{frame_filename}'.")
            else:
                logger.warning(f"Failed to read frame at index {current_frame}.")

            # Increment the frame counter by the frame_interval
            current_frame += frame_interval

            # Prevent infinite loop in case frame_interval is 0 or negative
            if frame_interval <= 0:
                logger.error(f"Invalid frame_interval={frame_interval}. It must be a positive integer.")
                break

        # Release the video capture object
        cap.release()
        logger.info(f"Frame extraction completed. Total frames extracted: {len(frames)}")

    except Exception as e:
        logger.error(f"An error occurred during frame extraction: {str(e)}")
        logger.debug(traceback.format_exc())
        raise e  # Re-raise the exception after logging

    return frames  # Return the list of extracted frames
