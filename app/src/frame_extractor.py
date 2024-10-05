# frame_extractor.py

import cv2
import logging
import traceback

# Setup logging for the frame extractor module
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture general information
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)
logger = logging.getLogger(__name__)

def extract_frames_opencv(video_path, max_frames=10, frame_interval=None):
    """
    Extracts frames from a video at regular intervals using OpenCV.

    Parameters:
        video_path (str): Path to the input video file.
        max_frames (int): Maximum number of frames to extract from the video.
        frame_interval (int, optional): Specific interval between frames to extract.
                                        If None, it is automatically calculated based on max_frames.

    Returns:
        List of frames in RGB format. Each frame is a NumPy array.
    """
    logger.info(f"Starting frame extraction from '{video_path}' with max_frames={max_frames} and frame_interval={frame_interval}")
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

        # Calculate frame_interval if not provided
        if frame_interval is None:
            frame_interval = max(total_frames // max_frames, 1)  # Ensure at least interval of 1
            logger.info(f"Calculated frame_interval based on max_frames: {frame_interval}")
        else:
            logger.info(f"Using provided frame_interval: {frame_interval}")

        current_frame = 0  # Frame counter

        # Loop through the video and extract frames at specified intervals
        while current_frame < total_frames and len(frames) < max_frames:
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
            else:
                logger.warning(f"Failed to read frame at index {current_frame}.")

            # Increment the frame counter by the frame_interval
            current_frame += frame_interval

        # Release the video capture object
        cap.release()
        logger.info(f"Frame extraction completed. Total frames extracted: {len(frames)}")

    except Exception as e:
        logger.error(f"An error occurred during frame extraction: {str(e)}")
        logger.debug(traceback.format_exc())
        raise e  # Re-raise the exception after logging

    return frames  # Return the list of extracted frames

