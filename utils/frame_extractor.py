import cv2
import os
import time  # Import the time module

def extract_frames_opencv(video_path, max_frames=10, frame_interval=None, trim_start=30, trim_end=30, frame_folder=None, target_resolution=(720, 480)):
    """
    Extracts frames from a video by dynamically adjusting the frame interval based on trimmed frames,
    resizes them to a specific resolution, and saves them into a specified output directory.

    Parameters:
        video_path (str): Path to the input video file.
        max_frames (int): Maximum number of frames to extract from the video.
        frame_interval (int, optional): Specific interval between frames to extract.
                                        If None, it is automatically calculated based on max_frames.
        trim_start (int): Number of frames to trim from the beginning of the video (to exclude intro).
        trim_end (int): Number of frames to trim from the end of the video (to exclude outro).
        output_dir (str, optional): Directory where extracted frames will be saved.
                                    If None, frames are not saved to disk.
        target_resolution (tuple): The target resolution for saved frames (width, height).

    Returns:
        List of frames in RGB format. Each frame is a NumPy array.
    """
    frames = []  # List to store the extracted frames

    try:
        # Start time tracking
        start_time = time.time()

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            raise ValueError("Invalid video file or cannot retrieve FPS and frame count.")

        # Ensure valid trimming range
        trim_start = min(trim_start, total_frames // 2)
        trim_end = min(trim_end, total_frames - trim_start)

        remaining_frames = total_frames - trim_start - trim_end
        if remaining_frames <= 0:
            return frames

        # Calculate frame_interval if not provided
        frame_interval = frame_interval or max(remaining_frames // max_frames, 1)

        # Adjust max_frames if frame_interval results in fewer frames
        max_frames = min(max_frames, (remaining_frames + frame_interval - 1) // frame_interval)  # Ceiling division


        # Frame extraction loop
        for i in range(max_frames):
            current_frame = trim_start + i * frame_interval
            if current_frame >= total_frames - trim_end:
                break

            # Set frame position and read the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Failed to read frame at position {current_frame}.")

            # Resize the frame to the target resolution
            frame_resized = cv2.resize(frame, target_resolution)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            # Save frame as image if output_dir is specified
            if frame_folder:
                frame_filename = os.path.join(frame_folder, f"frame_{current_frame}.jpg")
                cv2.imwrite(frame_filename, frame_resized)

        # Release the video capture object
        cap.release()

        # End time tracking
        end_time = time.time()

        # Print the total time taken
        print(f"Time taken to create frames: {end_time - start_time:.2f} seconds")

    except Exception as e:
        # Ensure proper release of the video capture object in case of error
        if 'cap' in locals():
            cap.release()
        raise RuntimeError(f"An error occurred during frame extraction: {str(e)}") from e

    return frames
