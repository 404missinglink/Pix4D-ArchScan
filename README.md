# PIX4D ARCHSCAN

A video to text summarisation tool using Pixtral

### Frame Extractor

```python
from frame_extractor import extract_frames_opencv

def main():
    video_path = "path_to_your_video.mp4"
    max_frames = 5
    frame_interval = 30  # Extract every 30th frame

    try:
        frames = extract_frames_opencv(video_path, max_frames=max_frames, frame_interval=frame_interval)
        print(f"Extracted {len(frames)} frames.")
        for idx, frame in enumerate(frames):
            # Display frame information
            height, width, channels = frame.shape
            print(f"Frame {idx}: {width}x{height}px, {channels} channels")
            # Further processing can be done here
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

```
