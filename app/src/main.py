import os
import gradio as gr
from mistralai import Mistral
from moviepy.editor import VideoFileClip
from PIL import Image
import tempfile
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Please set the API_KEY environment variable in the .env file.")

# Specify models
text_model = "mistral-large-latest"
vision_model = "pixtral-12b-2409"

# Initialize the Mistral client
client_text = Mistral(api_key=api_key)
client_vision = Mistral(api_key=api_key)

def extract_frames(video_path, max_frames=10):
    """
    Extracts frames from the video at regular intervals.
    """
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame_times = [i * duration / max_frames for i in range(max_frames)]
    frames = [clip.get_frame(t) for t in frame_times]
    return frames

def encode_image(image_path):
    """
    Encode the image to base64.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_video(video):
    """
    Processes the uploaded video and returns a JSON summary.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video.read())
        tmp_path = tmp.name
    
    try:
        frames = extract_frames(tmp_path, max_frames=10)
        
        frame_summaries = []
        for idx, frame in enumerate(frames):
            # Save frame as image temporarily
            frame_path = f"frame_{idx}.jpg"
            img = Image.fromarray(frame)
            img.save(frame_path)

            # Encode image to base64
            base64_image = encode_image(frame_path)
            if not base64_image:
                frame_summary = {"frame": idx, "error": "Failed to encode image."}
                frame_summaries.append(frame_summary)
                continue

            # Define the messages for the chat
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a summary of this frame."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Get the chat response with JSON format
            try:
                chat_response = client_vision.chat.complete(
                    model=vision_model,
                    messages=messages,
                    response_format={
                        "type": "json_object",
                    }
                )
                summary = chat_response.choices[0].message.content
                frame_summaries.append({"frame": idx, "summary": summary})
            except Exception as e:
                frame_summaries.append({"frame": idx, "error": str(e)})
            
            # Clean up the image file
            os.remove(frame_path)
        
        # Generate a JSON object summarizing all frames
        final_summary = {
            "video_summary": frame_summaries
        }
        return final_summary
    
    finally:
        os.remove(tmp_path)  # Clean up the video file

# Define Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Drone Footage"),
    outputs=gr.JSON(label="Live Summarization"),
    title="Drone Footage Summarizer",
    description="Upload drone video footage, and Pixtral will provide live summarizations of the content in JSON format."
)

if __name__ == "__main__":
    iface.launch()
