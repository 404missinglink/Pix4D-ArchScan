# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set the API_KEY environment variable in the .env file.")

# Model configurations
TEXT_MODEL = "mistral-large-latest"
PIXTRAL_API_URL = "http://127.0.0.1:5000/describe_image"  # Update if different
# VISION_MODEL = "pixtral-12b-2409"

# Rate limiting
RATE_LIMIT_SECONDS = 1.0

# Frame trimming constants
TRIM_START_FRAMES = 30
TRIM_END_FRAMES = 30

# Base directory to save frames and summaries
BASE_FRAMES_DIR = "frames"

# Hardcoded destination folder
UPLOAD_VIDEOS_FOLDER = "upload_videos"


# Ensure the base frames directory exists
os.makedirs(BASE_FRAMES_DIR, exist_ok=True)
