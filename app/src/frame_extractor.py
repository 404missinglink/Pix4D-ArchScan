import cv2
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_frames_opencv(video_path, max_frames=10, frame_interval=None):
    """
    Extracts frames from the video at regular intervals using OpenCV.

    Parameters:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to extract.
        frame_interval (int, optional): Interval between frames to extract.
                                        If None, it is calculated based on max_frames.

    Returns:
        List of frames in RGB format.
    """
    logger.info(f"Starting frame extraction from {video_path} with max_frames={max_frames} and frame_interval={frame_interval}")
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps else 0

        logger.info(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

        if frame_interval is None:
            frame_interval = max(total_frames // max_frames, 1)
            logger.info(f"Calculated frame_interval: {frame_interval}")

        current_frame = 0
        while current_frame < total_frames and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                logger.debug(f"Extracted frame {current_frame}")
            else:
                logger.warning(f"Failed to read frame at index {current_frame}")
            current_frame += frame_interval

        cap.release()
        logger.info(f"Extracted {len(frames)} frames.")
    except Exception as e:
        logger.error(f"Error during frame extraction: {str(e)}")
        logger.debug(traceback.format_exc())
        raise e
    return frames
