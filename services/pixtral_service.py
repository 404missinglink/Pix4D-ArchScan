import time
import requests
from threading import Lock
from PIL import Image
from utils.image_encoder import encode_image
import config.config as cf
import os

def process_with_pixtral(image_files, frame_folder_path, pixtral_response_path):
    """
    This function takes a list of image files and processes them using the Pixtral API.
    The responses are saved as text files in the provided response folder.

    Parameters:
    image_files (list): List of image file names.
    frame_folder_path (str): Path to the folder containing the image files.
    pixtral_response_path (str): Path to the folder where the responses will be saved.

    Returns:
    None
    """
    last_request_time = time.time()

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(frame_folder_path, image_file)
        img = Image.open(image_path)
        base64_image = encode_image(img)
        image_url = f"data:image/jpeg;base64,{base64_image}"

        rate_limit_lock = Lock()
        with rate_limit_lock:
            current_time = time.time()
            elapsed_time = current_time - last_request_time
            if elapsed_time < cf.RATE_LIMIT_SECONDS:
                sleep_time = cf.RATE_LIMIT_SECONDS - elapsed_time
                time.sleep(sleep_time)

            last_request_time = time.time()

        start_time = time.time()

        response = requests.post(
            cf.PIXTRAL_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "image_url": image_url,
                "prompt": cf.prompt
            }
        )

        end_time = time.time()
        latency = end_time - start_time

        if response.status_code != 200:
            raise ValueError(f"Pixtral API returned status code {response.status_code}: {response.text}")

        chat_response = response.json()
        summary = chat_response.get("description", "")

        if not summary.strip():
            raise ValueError("Pixtral API returned an empty summary.")

        print(f"Processed image {idx+1}/{len(image_files)} with Pixtral, latency: {latency:.2f} seconds")

        # Save response in a text file with the same name as the image (without .jpg)
        response_file_name = os.path.splitext(image_file)[0] + ".txt"
        response_file_path = os.path.join(pixtral_response_path, response_file_name)

        with open(response_file_path, 'w') as response_file:
            response_file.write(summary)

        print(f"Saved response for {image_file} to {response_file_path}")
