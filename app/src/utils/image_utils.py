# utils/image_utils.py

import base64
import tempfile
import os
from PIL import Image
import logging
import traceback

logger = logging.getLogger("DroneFootageSurveyor.utils.image_utils")

def encode_image(image: Image.Image) -> str:
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
