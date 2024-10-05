# logger.py

import logging

def setup_logger():
    """
    Configures and returns a logger.
    Ensures that handlers are added only once to prevent duplicate logs.
    """
    logger = logging.getLogger("DroneFootageSurveyor")
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # File handler
        file_handler = logging.FileHandler("app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()
