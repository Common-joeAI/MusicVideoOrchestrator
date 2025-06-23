# Music Video Orchestrator

A comprehensive application for generating AI music videos using ComfyUI, RIFE, and FFmpeg.

## Features

- Generate images from prompts using ComfyUI
- Interpolate frames using RIFE for smooth transitions
- Create videos with FFmpeg
- Embedded components for a seamless experience

## Setup

1. Make sure you have Python 3.10+ installed
2. Clone this repository
3. Run the setup script: `.\setup.ps1`
4. Activate the virtual environment: `.\venv\Scripts\Activate.ps1`
5. Run the application: `python main.py`

## Directory Structure

- `models/`: Model files for ComfyUI and RIFE
- `outputs/`: Generated images and videos
- `embedded/`: Embedded components (ComfyUI, RIFE, FFmpeg)
- `ui/`: User interface files
- `temp/`: Temporary files

## Dependencies

- ComfyUI: AI image generation
- RIFE: Frame interpolation
- FFmpeg: Video processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
