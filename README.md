# PIX4D ArchScan 🚁

**PIX4D ArchScan** is an advanced video-to-text summarisation tool designed to process drone footage of construction sites. It utilises a fine-tuned Pixtral model for frame descriptions and a Mistral Large agent for comprehensive assistance.

## Features

- **Video Processing**: Efficiently summarises video content to enhance construction site monitoring.
- **Dual-Agent Architecture**: Combines the strengths of a vision-language model (VLM) and a large language model (LLM) for superior summarisation accuracy.

## Prerequisites

To get started, ensure you have the following:

- Videos must be in **MP4** format.
- Python **3.10.14** or higher is required.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/404missinglink/Pix4D-ArchScan.git
   cd PIX4D-ArchScan
   ```

2. Navigate to the source directory:

   ```bash
   cd app/src
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the API endpoint and key:

   - Open `config.py` and set the Pixtral model endpoint:
     ```python
     PIXTRAL_API_URL = "http://127.0.0.1:5000/describe_image"  # Update if different
     ```

5. Create a `.env` file in the same directory with your API key:
   ```
   API_KEY="YOUR_API_KEY"
   ```

## Usage

To launch the Gradio UI, simply run:

```bash
python main.py
```

You can then upload your video files directly through the interface.

## Architecture

This version employs a finetuned pixtral VLM on a Nvidia H100 and a base mistral-large-latest model available on **Le Platforme**. The processing times are as follows:

- Each frame takes approximately **10 seconds** to process using the Pixtral model.
- The total time for generating the overall summary can take up to **34 seconds**, depending on the number of frames processed.

### Accessing the H100 API from Nebius

```
ssh -L 5000:127.0.0.1:5000 admin@195.242.23.219
```
