# PIX4D ARCHSCAN

A video to text summarisation tool using a finetuned Pixtral model for frame description and a Mistral Large 2 Agent for assistance.

## Prerequisites

- Video's must be in `MP4` format
- Python 3.10.14

## Architecture

This version uses the base Mistral models on Le Platforme only, each frame takes approx 10 seconds to process by pixtral and the overall summary can take up to 34 seconds to process, the overall summary time is dependant on the amount of frames processed.
