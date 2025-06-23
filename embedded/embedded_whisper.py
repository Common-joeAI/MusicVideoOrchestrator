import os
import subprocess
import json
from typing import List, Dict, Any, Optional


class EmbeddedWhisper:
    """Class to embed whisper.cpp functionality for audio transcription."""

    def __init__(self, whisper_exe_path: str, model_path: str):
        self.whisper_exe_path = os.path.abspath(whisper_exe_path)
        self.model_path = os.path.abspath(model_path)

        if not os.path.exists(self.whisper_exe_path):
            raise FileNotFoundError(f"whisper.cpp executable not found at {self.whisper_exe_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")

        print(f"Whisper initialized with executable: {self.whisper_exe_path}")
        print(f"Whisper model: {self.model_path}")

    def transcribe_audio(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes an audio file and returns a list of segments with timestamps.
        Identifies gaps (non-lyric sections) as well.
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Use -oj (output JSON) and -otv (output timestamps verbose)
        # -l auto (language auto-detection)
        cmd = [
            self.whisper_exe_path,
            "-m", self.model_path,
            "-f", audio_file_path,
            "-oj", "-otv", "-l", "auto"
        ]

        print(f"Running Whisper command: {' '.join(cmd)}")
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'  # Ensure correct encoding for output
            )

            # Whisper.cpp outputs JSON to stdout
            output_json = json.loads(process.stdout)
            segments = output_json.get("transcription", [])

            # Process segments to include gaps
            processed_segments = []
            last_end_time = 0.0

            for segment in segments:
                start_time = segment["t0"] / 1000.0  # Convert ms to seconds
                end_time = segment["t1"] / 1000.0  # Convert ms to seconds
                text = segment["text"].strip()

                # Add gap if there's a significant one
                if start_time - last_end_time > 0.5:  # Define a threshold for a "gap"
                    processed_segments.append({
                        'start': last_end_time,
                        'end': start_time,
                        'type': 'gap',
                        'text': ''  # No lyrics in a gap
                    })

                processed_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'type': 'lyric',
                    'text': text
                })
                last_end_time = end_time

            return processed_segments

        except subprocess.CalledProcessError as e:
            print(f"Whisper.cpp error: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            print(f"Failed to parse Whisper.cpp JSON output: {e}")
            print(f"Raw output: {process.stdout}")
            raise
        except Exception as e:
            print(f"Error running Whisper: {str(e)}")
            raise

