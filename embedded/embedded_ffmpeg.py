import os
import subprocess
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Tuple


class EmbeddedFFmpeg:
    """Class to embed FFmpeg functionality directly in the application"""

    def __init__(self, ffmpeg_path: str):
        self.ffmpeg_path = os.path.abspath(ffmpeg_path)
        self.ffmpeg_exe = os.path.join(self.ffmpeg_path, "ffmpeg.exe")
        self.ffprobe_exe = os.path.join(self.ffmpeg_path, "ffprobe.exe")

        # Verify FFmpeg exists
        if not os.path.exists(self.ffmpeg_exe):
            raise FileNotFoundError(f"FFmpeg executable not found at {self.ffmpeg_exe}")

        # Verify FFprobe exists
        if not os.path.exists(self.ffprobe_exe):
            raise FileNotFoundError(f"FFprobe executable not found at {self.ffprobe_exe}")

        print(f"FFmpeg initialized with path: {self.ffmpeg_path}")
        print(f"FFmpeg executable: {self.ffmpeg_exe}")
        print(f"FFprobe executable: {self.ffprobe_exe}")

    def create_video_from_frames(self,
                                 frames_dir: str,
                                 output_file: str,
                                 fps: int = 30,
                                 resolution: Tuple[int, int] = (1920, 1080),
                                 audio_file: Optional[str] = None,
                                 crf: int = 18,
                                 preset: str = "slow") -> bool:
        """Create a video from a directory of frames"""
        # Build FFmpeg command
        width, height = resolution

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Check if frames directory exists and has frames
        if not os.path.exists(frames_dir):
            print(f"Frames directory not found: {frames_dir}")
            return False

        # Check for frame files
        frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")]
        if not frame_files:
            print(f"No frame files found in directory: {frames_dir}")
            return False

        print(f"Found {len(frame_files)} frame files in {frames_dir}")

        # Determine the frame filename pattern
        # Check if frames are named with 6 digits (frame_000001.png) or without leading zeros
        has_leading_zeros = any(f for f in frame_files if f.startswith("frame_0"))
        frame_pattern = "frame_%06d.png" if has_leading_zeros else "frame_%d.png"

        cmd = [
            self.ffmpeg_exe,
            "-y",  # Overwrite output file if it exists
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, frame_pattern),
            "-vf", f"scale={width}:{height}:flags=lanczos",
        ]

        if audio_file and os.path.exists(audio_file):
            cmd.extend([
                "-i", audio_file,
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",  # End when the shortest input stream ends
                output_file
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                output_file
            ])

        # Run FFmpeg
        try:
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Monitor FFmpeg output
            for line in process.stdout:
                print(f"FFmpeg: {line.strip()}")

            process.wait()

            if process.returncode != 0:
                print(f"FFmpeg error: Return code {process.returncode}")
                return False

            print(f"Video created successfully: {output_file}")
            return True
        except Exception as e:
            print(f"Error running FFmpeg: {str(e)}")
            return False

    def add_intro_video(self,
                        intro_file: str,
                        main_file: str,
                        output_file: str,
                        transition_duration: float = 1.0) -> bool:
        """Add an intro video to the main video with a fade transition"""
        if not os.path.exists(intro_file):
            print(f"Intro file not found: {intro_file}")
            return False

        if not os.path.exists(main_file):
            print(f"Main file not found: {main_file}")
            return False

        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        try:
            # Get intro duration
            intro_duration = self.get_video_duration(intro_file)
            if intro_duration is None:
                print("Could not determine intro duration")
                return False

            print(f"Intro video duration: {intro_duration} seconds")

            # Calculate transition start time
            transition_start = max(0, intro_duration - transition_duration)

            # Create the fade out effect for the intro
            intro_fade_file = os.path.join(temp_dir, "intro_fade.mp4")
            cmd_fade = [
                self.ffmpeg_exe,
                "-y",
                "-i", intro_file,
                "-vf", f"fade=t=out:st={transition_start}:d={transition_duration}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "copy",
                intro_fade_file
            ]

            print(f"Creating fade-out intro: {' '.join(cmd_fade)}")
            subprocess.run(cmd_fade, check=True)

            # Create the fade in effect for the main video
            main_fade_file = os.path.join(temp_dir, "main_fade.mp4")
            cmd_fade_main = [
                self.ffmpeg_exe,
                "-y",
                "-i", main_file,
                "-vf", f"fade=t=in:st=0:d={transition_duration}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "copy",
                main_fade_file
            ]

            print(f"Creating fade-in main video: {' '.join(cmd_fade_main)}")
            subprocess.run(cmd_fade_main, check=True)

            # Create a file list for concatenation
            file_list = os.path.join(temp_dir, "file_list.txt")
            with open(file_list, "w") as f:
                # Replace backslashes with forward slashes for FFmpeg
                intro_path = intro_fade_file.replace('\\', '/')
                main_path = main_fade_file.replace('\\', '/')
                f.write(f"file '{intro_path}'\n")
                f.write(f"file '{main_path}'\n")

            # Concatenate the videos
            cmd_concat = [
                self.ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", file_list,
                "-c", "copy",
                output_file
            ]

            print(f"Concatenating videos: {' '.join(cmd_concat)}")
            subprocess.run(cmd_concat, check=True)

            print(f"Video with intro created successfully: {output_file}")
            return True
        except Exception as e:
            print(f"Error adding intro video: {str(e)}")
            return False
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    def get_video_duration(self, video_file: str) -> Optional[float]:
        """Get the duration of a video file in seconds"""
        cmd = [
            self.ffprobe_exe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_file
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"Error getting video duration: {str(e)}")
            return None

    def extract_audio(self, video_file: str, output_file: str) -> bool:
        """Extract audio from a video file"""
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i", video_file,
            "-vn",  # No video
            "-acodec", "copy",  # Copy audio codec
            output_file
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Audio extracted successfully: {output_file}")
            return True
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return False

    def create_preview(self, video_file: str, output_file: str, width: int = 640, height: int = 360) -> bool:
        """Create a low-resolution preview of a video"""
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i", video_file,
            "-vf", f"scale={width}:{height}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            output_file
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Preview created successfully: {output_file}")
            return True
        except Exception as e:
            print(f"Error creating preview: {str(e)}")
            return False
