import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from utils.image_encoder import encode_image
import config.config as cf
import os

def process_image(image_file, frame_folder_path):
    """
    This function processes a single image file using the Pixtral API.
    
    Parameters:
    image_file (str): Image file name.
    frame_folder_path (str): Path to the folder containing the image files.
    
    Returns:
    str: The summary response from the Pixtral API.
    """
    image_path = os.path.join(frame_folder_path, image_file)
    img = Image.open(image_path)
    base64_image = encode_image(img)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    response = requests.post(
        cf.PIXTRAL_API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "image_url": image_url,
            "prompt": cf.prompt
        }
    )

    if response.status_code != 200:
        raise ValueError(f"Pixtral API returned status code {response.status_code}: {response.text}")

    chat_response = response.json()
    summary = chat_response.get("description", "")

    if not summary.strip():
        raise ValueError("Pixtral API returned an empty summary.")

    return image_file, summary

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
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(process_image, image_file, frame_folder_path): image_file for image_file in image_files}

        for future in as_completed(future_to_image):
            image_file = future_to_image[future]
            try:
                summary = future.result()
                # Save response in a text file with the same name as the image (without .jpg)
                response_file_name = os.path.splitext(image_file)[0] + ".txt"
                response_file_path = os.path.join(pixtral_response_path, response_file_name)

                with open(response_file_path, 'w') as response_file:
                    response_file.write(summary)

                print(f"Saved response for {image_file} to {response_file_path}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

