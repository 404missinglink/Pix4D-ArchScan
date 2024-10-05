# utils/video_utils.py

import tempfile
import os
import logging
import traceback

logger = logging.getLogger("DroneFootageSurveyor.utils.video_utils")

def save_temp_video(video_path: str) -> str:
    """
    Saves the uploaded video to a temporary file.

    Parameters:
        video_path (str): Path to the uploaded video.

    Returns:
        str: Path to the temporary video file.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            with open(video_path, "rb") as f:
                tmp.write(f.read())
            tmp_path = tmp.name
            logger.info(f"Temporary video saved at {tmp_path}")
            return tmp_path
    except Exception as e:
        logger.error(f"Failed to save temporary video file: {str(e)}")
        logger.debug(traceback.format_exc())
        raise
