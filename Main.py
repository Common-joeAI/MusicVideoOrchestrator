import sys
import os
import threading
import json
import csv
import re
import cv2
import numpy as np
import shutil
import time
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QTextEdit, QLabel, QProgressBar, QLineEdit,
                             QTabWidget, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QComboBox, QMessageBox, QScrollBar)
from PyQt5.QtCore import Qt, QThread, QSettings, pyqtSignal, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

# Add embedded modules to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDED_DIR = os.path.join(SCRIPT_DIR, "embedded")
sys.path.append(EMBEDDED_DIR)

# Import embedded modules
try:
    from embedded_comfy import EmbeddedComfyUI
    from embedded_rife import EmbeddedRIFE
    from embedded_ffmpeg import EmbeddedFFmpeg
    from embedded_whisper import EmbeddedWhisper  # New import
    from embedded_llama import EmbeddedLlama  # New import
except ImportError as e:
    print(f"Error importing embedded modules: {e}")
    print("Make sure the embedded modules are properly installed in the 'embedded' directory.")
    sys.exit(1)


# Worker thread class - Defined at the top level
class WorkerThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    complete_signal = pyqtSignal(dict)

    def __init__(self, task_type, params=None):
        super().__init__()
        self.task_type = task_type
        self.params = params or {}
        self.running = True

    def run(self):
        try:
            if self.task_type == "generate_images":
                self.generate_images()
            elif self.task_type == "interpolate_frames":
                self.interpolate_frames()
            elif self.task_type == "create_video":
                self.create_video()
            elif self.task_type == "generate_llm_prompts":  # New task type
                self.generate_llm_prompts()
        except Exception as e:
            import traceback
            self.update_signal.emit(f"Error in {self.task_type}: {str(e)}")
            self.update_signal.emit(traceback.format_exc())
            self.complete_signal.emit({"success": False, "error": str(e)})

    def generate_images(self):
        """Generate images using embedded ComfyUI"""
        workflow_file = self.params.get("workflow_file")
        prompts_file = self.params.get("prompts_file")
        output_dir = self.params.get("output_dir")
        comfy = self.params.get("comfy")  # ComfyUI instance

        if not comfy:
            self.update_signal.emit("ComfyUI not initialized")
            self.complete_signal.emit({"success": False, "error": "ComfyUI not initialized"})
            return

        os.makedirs(output_dir, exist_ok=True)

        with open(workflow_file, 'r') as f:
            workflow_data = json.load(f)

        prompts = []
        with open(prompts_file, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                prompts.append({
                    'start': float(row['Start']),
                    'end': float(row['End']),
                    'prompt': row['Prompt']
                })

        self.update_signal.emit(f"Processing {len(prompts)} prompts...")

        for i, prompt_data in enumerate(prompts):
            if not self.running:
                self.update_signal.emit("Operation cancelled")
                self.complete_signal.emit({"success": False, "message": "Operation cancelled"})
                return

            prompt_text = prompt_data['prompt']
            timestamp = f"{prompt_data['start']:.2f}-{prompt_data['end']:.2f}"

            self.update_signal.emit(
                f"[{i + 1}/{len(prompts)}] Processing prompt at {timestamp}: {prompt_text[:50]}...")

            workflow = workflow_data.copy()
            workflow["40"]["inputs"]["text"] = prompt_text

            try:
                prompt_id = comfy.execute_workflow(workflow)
                self.update_signal.emit("  Executing workflow...")

                while True:
                    if not self.running:
                        self.update_signal.emit("Operation cancelled")
                        self.complete_signal.emit({"success": False, "message": "Operation cancelled"})
                        return

                    completed, progress, result = comfy.get_execution_status(prompt_id)
                    self.progress_signal.emit(int(progress * 100))

                    if completed:
                        break
                    time.sleep(0.1)

                if "error" in result:
                    self.update_signal.emit(f"  Error: {result['error']}")
                    continue

                image_path = comfy.get_image_path(result)
                if image_path:
                    local_filename = f"{output_dir}/frame_{timestamp.replace('.', '_')}.png"
                    shutil.copy2(image_path, local_filename)
                    self.update_signal.emit(f"  Saved image to {local_filename}")
                else:
                    self.update_signal.emit("  No image generated")

            except Exception as e:
                self.update_signal.emit(f"  Error: {str(e)}")

            overall_progress = int((i + 1) / len(prompts) * 100)
            self.progress_signal.emit(overall_progress)

        self.update_signal.emit("All prompts processed!")
        self.complete_signal.emit({
            "success": True,
            "output_dir": output_dir,
            "frame_count": len(prompts)
        })

    def interpolate_frames(self):
        """Interpolate frames using OpenCV or RIFE"""
        input_dir = self.params.get("input_dir")
        temp_dir = self.params.get("temp_dir")
        output_dir = self.params.get("output_dir")
        fps = self.params.get("fps", 30)
        method = self.params.get("method", "OpenCV")
        rife = self.params.get("rife")  # RIFE instance

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.update_signal.emit("Analyzing and sorting frames...")

        files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

        frame_data = []
        for file in files:
            match = re.search(r'frame_(\d+)_(\d+)-(\d+)_(\d+)', file)
            if match:
                start_sec = float(f"{match.group(1)}.{match.group(2)}")
                end_sec = float(f"{match.group(3)}.{match.group(4)}")
                frame_data.append({
                    'filename': file,
                    'start_time': start_sec,
                    'end_time': end_sec,
                    'duration': end_sec - start_sec
                })

        frame_data.sort(key=lambda x: x['start_time'])

        self.update_signal.emit(f"Found {len(frame_data)} keyframes")

        for i in range(1, len(frame_data)):
            prev_end = frame_data[i - 1]['end_time']
            curr_start = frame_data[i]['start_time']

            if abs(curr_start - prev_end) > 0.1:
                if curr_start > prev_end:
                    self.update_signal.emit(
                        f"Warning: Gap detected between {frame_data[i - 1]['filename']} and {frame_data[i]['filename']} ({prev_end} to {curr_start})")
                else:
                    self.update_signal.emit(
                        f"Warning: Overlap detected between {frame_data[i - 1]['filename']} and {frame_data[i]['filename']} ({prev_end} to {curr_start})")

        self.update_signal.emit("Preparing frames in sequential order...")

        for i, frame in enumerate(frame_data):
            if not self.running:
                self.update_signal.emit("Operation cancelled")
                self.complete_signal.emit({"success": False, "message": "Operation cancelled"})
                return

            src_path = os.path.join(input_dir, frame['filename'])
            dst_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            shutil.copy2(src_path, dst_path)
            frame['ordered_path'] = dst_path

            progress = int((i + 1) / len(frame_data) * 100)
            self.progress_signal.emit(progress)

        self.update_signal.emit(f"Interpolating frames using {method}...")

        frame_index = 0

        for i in range(len(frame_data) - 1):
            if not self.running:
                self.update_signal.emit("Operation cancelled")
                self.complete_signal.emit({"success": False, "message": "Operation cancelled"})
                return

            current_frame = cv2.imread(frame_data[i]['ordered_path'])
            next_frame = cv2.imread(frame_data[i + 1]['ordered_path'])

            duration = frame_data[i + 1]['start_time'] - frame_data[i]['start_time']
            num_frames = max(1, int(duration * fps))

            output_path = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
            cv2.imwrite(output_path, current_frame)
            frame_index += 1

            if num_frames <= 1:
                continue

            if method == "RIFE" and rife:
                try:
                    interpolated_frames = rife.interpolate_frames(current_frame, next_frame, num_frames - 1)

                    for j, frame in enumerate(interpolated_frames):
                        output_path = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
                        cv2.imwrite(output_path, frame)
                        frame_index += 1
                except Exception as e:
                    self.update_signal.emit(f"  RIFE interpolation failed: {str(e)}. Falling back to OpenCV.")
                    for j in range(1, num_frames):
                        alpha = j / num_frames
                        interpolated = cv2.addWeighted(current_frame, 1 - alpha, next_frame, alpha, 0)
                        output_path = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
                        cv2.imwrite(output_path, interpolated)
                        frame_index += 1
            else:
                for j in range(1, num_frames):
                    alpha = j / num_frames
                    interpolated = cv2.addWeighted(current_frame, 1 - alpha, next_frame, alpha, 0)
                    output_path = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
                    cv2.imwrite(output_path, interpolated)
                    frame_index += 1

            progress = int((i + 1) / (len(frame_data) - 1) * 100)
            self.progress_signal.emit(progress)

        output_path = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
        cv2.imwrite(output_path, cv2.imread(frame_data[-1]['ordered_path']))
        frame_index += 1

        self.update_signal.emit(f"Generated {frame_index} frames after interpolation")
        self.complete_signal.emit({
            "success": True,
            "output_dir": output_dir,
            "frame_count": frame_index
        })

    def create_video(self):
        """Create a video using FFmpeg"""
        frames_dir = self.params.get("frames_dir")
        audio_file = self.params.get("audio_file")
        intro_file = self.params.get("intro_file")
        output_file = self.params.get("output_file")
        fps = self.params.get("fps", 30)
        resolution = self.params.get("resolution", (1920, 1080))
        crf = self.params.get("crf", 18)
        preset = self.params.get("preset", "slow")
        ffmpeg = self.params.get("ffmpeg")  # FFmpeg instance

        if not ffmpeg:
            self.update_signal.emit("FFmpeg not initialized")
            self.complete_signal.emit({"success": False, "error": "FFmpeg not initialized"})
            return

        self.update_signal.emit(f"Creating video from frames at {resolution[0]}x{resolution[1]}, {fps} FPS...")

        main_video_file = output_file
        if intro_file:
            main_video_file = output_file.replace(".mp4", "_main.mp4")

        success = ffmpeg.create_video_from_frames(
            frames_dir=frames_dir,
            output_file=main_video_file,
            fps=fps,
            resolution=resolution,
            audio_file=audio_file,
            crf=crf,
            preset=preset
        )

        if not success:
            self.update_signal.emit("Failed to create video from frames")
            self.complete_signal.emit({"success": False, "error": "Failed to create video from frames"})
            return

        self.update_signal.emit("Video created successfully")

        if intro_file and os.path.exists(intro_file):
            self.update_signal.emit("Adding intro video...")

            success = ffmpeg.add_intro_video(
                intro_file=intro_file,
                main_file=main_video_file,
                output_file=output_file,
                transition_duration=1.0
            )

            if not success:
                self.update_signal.emit("Failed to add intro video")
                self.complete_signal.emit({"success": False, "error": "Failed to add intro video"})
                return

            self.update_signal.emit("Intro added successfully")

            try:
                os.remove(main_video_file)
            except:
                pass

        self.update_signal.emit(f"Video creation complete: {output_file}")
        self.complete_signal.emit({
            "success": True,
            "output_file": output_file
        })

    def generate_llm_prompts(self):
        """Generates image prompts using Whisper for transcription and Llama for LLM."""
        audio_file = self.params.get("audio_file")
        song_description = self.params.get("song_description")
        whisper = self.params.get("whisper")
        llama = self.params.get("llama")

        if not whisper:
            self.update_signal.emit("Whisper not initialized.")
            self.complete_signal.emit({"success": False, "error": "Whisper not initialized."})
            return
        if not llama:
            self.update_signal.emit("Llama not initialized.")
            self.complete_signal.emit({"success": False, "error": "Llama not initialized."})
            return

        try:
            self.update_signal.emit("Transcribing audio with Whisper...")
            audio_segments = whisper.transcribe_audio(audio_file)
            self.update_signal.emit(f"Whisper transcription complete. Found {len(audio_segments)} segments.")

            generated_prompts = []
            total_segments = len(audio_segments)

            for i, segment in enumerate(audio_segments):
                if not self.running:
                    self.update_signal.emit("Operation cancelled.")
                    self.complete_signal.emit({"success": False, "message": "Operation cancelled."})
                    return

                segment_text = segment['text']
                segment_type = segment['type']
                start_time = segment['start']
                end_time = segment['end']

                # Base prompt instruction for the LLM
                prompt_instruction = (
                    "Generate a concise, visually descriptive image prompt (max 15 words) for a music video segment. "
                    "The song is an EDM track inspired by the Bible, with the overall theme: "
                    f"'{song_description}'.\n"
                )

                if segment_type == 'lyric':
                    prompt_context = (
                        f"This segment has lyrics: '{segment_text}'. "
                        "Focus on abstract, spiritual, or energetic visuals related to the lyrics and song theme."
                    )
                else:  # type == 'gap'
                    prompt_context = (
                        "This is an instrumental section. "
                        "Focus on abstract, energetic, or transitional visuals that fit the song's theme and flow from the previous scene."
                    )

                full_prompt = prompt_instruction + prompt_context

                self.update_signal.emit(
                    f"[{i + 1}/{total_segments}] Generating prompt for {segment_type} section ({start_time:.2f}-{end_time:.2f})...")

                # Call Llama to generate the prompt
                llm_generated_text = llama.generate_text(full_prompt, max_tokens=50, temperature=0.7)

                generated_prompts.append({
                    'Start': start_time,
                    'End': end_time,
                    'Prompt': llm_generated_text.strip()
                })
                self.update_signal.emit(f"  Generated: {llm_generated_text.strip()}")

                self.progress_signal.emit(int((i + 1) / total_segments * 100))
                time.sleep(0.1)

            # Save generated prompts to a CSV file
            output_filename = os.path.join(os.path.dirname(audio_file), f"{Path(audio_file).stem}_llm_prompts.csv")
            with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Start', 'End', 'Prompt']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(generated_prompts)

            self.update_signal.emit(f"All prompts generated and saved to {output_filename}")
            self.complete_signal.emit({"success": True, "output_file": output_filename})

        except Exception as e:
            self.update_signal.emit(f"Error during prompt generation: {str(e)}")
            self.complete_signal.emit({"success": False, "error": str(e)})


# Main Application Class
class MusicVideoOrchestrator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Video Orchestrator")
        self.setGeometry(100, 100, 1200, 800)

        # State variables
        self.project_dir = ""
        self.workflow_file = ""
        self.prompts_file = ""
        self.audio_file = ""
        self.intro_file = ""
        self.images_dir = ""
        self.interpolated_dir = ""
        self.output_video = ""

        # Embedded components instances
        self.comfy = None
        self.rife = None
        self.ffmpeg = None
        self.whisper = None  # New instance
        self.llama = None  # New instance

        # Settings
        self.settings = QSettings("ElectricChristian", "MusicVideoOrchestrator")

        # Initialize UI
        self.init_ui()

        # Load saved settings
        self.load_settings()

        # Worker thread
        self.worker = None

    def init_ui(self):
        # Main layout with tabs
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # Create tabs
        self.tabs = QTabWidget()
        self.project_tab = QWidget()
        self.generate_tab = QWidget()
        self.interpolate_tab = QWidget()
        self.video_tab = QWidget()
        self.comfyui_tab = QWidget()
        self.prompt_generation_tab = QWidget()  # New tab

        self.tabs.addTab(self.project_tab, "Project Setup")
        self.tabs.addTab(self.prompt_generation_tab, "Generate Prompts")  # New tab position
        self.tabs.addTab(self.generate_tab, "Generate Images")
        self.tabs.addTab(self.interpolate_tab, "Interpolate Frames")
        self.tabs.addTab(self.video_tab, "Create Video")
        self.tabs.addTab(self.comfyui_tab, "ComfyUI Interface")

        # Setup each tab
        self.setup_project_tab()
        self.setup_prompt_generation_tab()  # Setup new tab
        self.setup_generate_tab()
        self.setup_interpolate_tab()
        self.setup_video_tab()
        self.setup_comfyui_tab()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Console output
        console_label = QLabel("Console Output:")
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)

        # Add tabs and console to main layout
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(console_label)
        self.main_layout.addWidget(self.console_output)

        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

    def setup_project_tab(self):
        layout = QVBoxLayout()

        # Project directory
        project_group = QGroupBox("Project Directory")
        project_layout = QHBoxLayout()
        self.project_path_label = QLabel("No directory selected")
        project_button = QPushButton("Select Project Directory")
        project_button.clicked.connect(self.select_project_dir)
        project_layout.addWidget(self.project_path_label, 1)
        project_layout.addWidget(project_button)
        project_group.setLayout(project_layout)

        # Files group
        files_group = QGroupBox("Project Files")
        files_layout = QFormLayout()

        # Workflow file
        workflow_layout = QHBoxLayout()
        self.workflow_path_label = QLabel("No file selected")
        workflow_button = QPushButton("Select")
        workflow_button.clicked.connect(self.select_workflow_file)
        workflow_layout.addWidget(self.workflow_path_label, 1)
        workflow_layout.addWidget(workflow_button)
        files_layout.addRow("ComfyUI Workflow:", workflow_layout)

        # Prompts file (this will be the input for image generation, or output of prompt generation)
        prompts_layout = QHBoxLayout()
        self.prompts_path_label = QLabel("No file selected")
        prompts_button = QPushButton("Select")
        prompts_button.clicked.connect(self.select_prompts_file)
        prompts_layout.addWidget(self.prompts_path_label, 1)
        prompts_layout.addWidget(prompts_button)
        files_layout.addRow("Prompts CSV:", prompts_layout)

        # Audio file (input for prompt generation)
        audio_layout = QHBoxLayout()
        self.audio_path_label = QLabel("No file selected")
        audio_button = QPushButton("Select")
        audio_button.clicked.connect(self.select_audio_file)
        audio_layout.addWidget(self.audio_path_label, 1)
        audio_layout.addWidget(audio_button)
        files_layout.addRow("Audio File:", audio_layout)

        # Intro video (optional)
        intro_layout = QHBoxLayout()
        self.intro_path_label = QLabel("No file selected (optional)")
        intro_button = QPushButton("Select")
        intro_button.clicked.connect(self.select_intro_file)
        intro_layout.addWidget(self.intro_path_label, 1)
        intro_layout.addWidget(intro_button)
        files_layout.addRow("Intro Video (optional):", intro_layout)

        files_group.setLayout(files_layout)

        # ComfyUI integration settings
        comfy_group = QGroupBox("ComfyUI Integration")
        comfy_layout = QFormLayout()

        self.comfy_models_dir = QLineEdit()
        models_browse = QPushButton("Browse")
        models_browse.clicked.connect(self.select_models_dir)

        models_layout = QHBoxLayout()
        models_layout.addWidget(self.comfy_models_dir, 1)
        models_layout.addWidget(models_browse)

        comfy_layout.addRow("Models Directory:", models_layout)

        self.comfy_output_dir = QLineEdit()
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self.select_comfy_output_dir)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.comfy_output_dir, 1)
        output_layout.addWidget(output_browse)

        comfy_layout.addRow("ComfyUI Output:", output_layout)

        self.comfy_server_input = QLineEdit("127.0.0.1:8188")
        comfy_layout.addRow("ComfyUI Server:", self.comfy_server_input)

        self.comfy_init_button = QPushButton("Initialize ComfyUI")
        self.comfy_init_button.clicked.connect(self.initialize_comfyui)
        comfy_layout.addRow("", self.comfy_init_button)

        comfy_group.setLayout(comfy_layout)

        # RIFE & Whisper integration settings
        rife_whisper_group = QGroupBox("RIFE & Whisper Integration")  # Renamed group
        rife_whisper_layout = QFormLayout()

        self.rife_models_dir = QLineEdit()
        rife_models_browse = QPushButton("Browse")
        rife_models_browse.clicked.connect(self.select_rife_models_dir)

        rife_models_layout = QHBoxLayout()
        rife_models_layout.addWidget(self.rife_models_dir, 1)
        rife_models_layout.addWidget(rife_models_browse)

        rife_whisper_layout.addRow("RIFE Models Directory:", rife_models_layout)

        self.rife_init_button = QPushButton("Initialize RIFE")
        self.rife_init_button.clicked.connect(self.initialize_rife)
        rife_whisper_layout.addRow("", self.rife_init_button)

        self.whisper_exe_path = QLineEdit()  # New Whisper path
        whisper_exe_browse = QPushButton("Browse")
        whisper_exe_browse.clicked.connect(self.select_whisper_exe_path)
        whisper_exe_layout = QHBoxLayout()
        whisper_exe_layout.addWidget(self.whisper_exe_path, 1)
        whisper_exe_layout.addWidget(whisper_exe_browse)
        rife_whisper_layout.addRow("Whisper.cpp Executable:", whisper_exe_layout)

        self.whisper_model_path = QLineEdit()  # New Whisper model path
        whisper_model_browse = QPushButton("Browse")
        whisper_model_browse.clicked.connect(self.select_whisper_model_path)
        whisper_model_layout = QHBoxLayout()
        whisper_model_layout.addWidget(self.whisper_model_path, 1)
        whisper_model_layout.addWidget(whisper_model_browse)
        rife_whisper_layout.addRow("Whisper Model:", whisper_model_layout)

        self.whisper_init_button = QPushButton("Initialize Whisper")  # New Whisper init button
        self.whisper_init_button.clicked.connect(self.initialize_whisper)
        rife_whisper_layout.addRow("", self.whisper_init_button)

        rife_whisper_group.setLayout(rife_whisper_layout)

        # Llama integration settings
        llama_group = QGroupBox("LLM Prompt Generation (Llama.cpp)")  # New group
        llama_layout = QFormLayout()

        self.llama_exe_path = QLineEdit()
        llama_exe_browse = QPushButton("Browse")
        llama_exe_browse.clicked.connect(self.select_llama_exe_path)
        llama_exe_layout = QHBoxLayout()
        llama_exe_layout.addWidget(self.llama_exe_path, 1)
        llama_exe_layout.addWidget(llama_exe_browse)
        llama_layout.addRow("Llama.cpp Executable:", llama_exe_layout)

        self.llama_model_path = QLineEdit()
        llama_model_browse = QPushButton("Browse")
        llama_model_browse.clicked.connect(self.select_llama_model_path)
        llama_model_layout = QHBoxLayout()
        llama_model_layout.addWidget(self.llama_model_path, 1)
        llama_model_layout.addWidget(llama_model_browse)
        llama_layout.addRow("Llama Model:", llama_model_layout)

        self.llama_init_button = QPushButton("Initialize Llama")
        self.llama_init_button.clicked.connect(self.initialize_llama)
        llama_layout.addRow("", self.llama_init_button)

        self.song_description_input = QLineEdit()  # New song description input
        llama_layout.addRow("Song Description:", self.song_description_input)

        llama_group.setLayout(llama_layout)

        # FFmpeg integration settings
        ffmpeg_group = QGroupBox("FFmpeg Integration")
        ffmpeg_layout = QFormLayout()

        self.ffmpeg_path = QLineEdit()
        ffmpeg_browse = QPushButton("Browse")
        ffmpeg_browse.clicked.connect(self.select_ffmpeg_path)

        ffmpeg_path_layout = QHBoxLayout()
        ffmpeg_path_layout.addWidget(self.ffmpeg_path, 1)
        ffmpeg_path_layout.addWidget(ffmpeg_browse)

        ffmpeg_layout.addRow("FFmpeg Directory:", ffmpeg_path_layout)

        self.ffmpeg_init_button = QPushButton("Initialize FFmpeg")
        self.ffmpeg_init_button.clicked.connect(self.initialize_ffmpeg)
        ffmpeg_layout.addRow("", self.ffmpeg_init_button)

        ffmpeg_group.setLayout(ffmpeg_layout)

        # Add all groups to layout
        layout.addWidget(project_group)
        layout.addWidget(files_group)
        layout.addWidget(comfy_group)
        layout.addWidget(rife_whisper_group)  # Use renamed group
        layout.addWidget(llama_group)  # Add Llama group
        layout.addWidget(ffmpeg_group)
        layout.addStretch()

        self.project_tab.setLayout(layout)

    def setup_prompt_generation_tab(self):  # New tab setup method
        layout = QVBoxLayout()

        prompt_gen_group = QGroupBox("Generate Prompts from Audio")
        prompt_gen_layout = QFormLayout()

        self.generated_prompts_output = QTextEdit()
        self.generated_prompts_output.setReadOnly(True)
        prompt_gen_layout.addRow("Generated Prompts Preview:", self.generated_prompts_output)

        self.generate_prompts_button = QPushButton("Generate Prompts")
        self.generate_prompts_button.clicked.connect(self.start_prompt_generation)
        prompt_gen_layout.addRow("", self.generate_prompts_button)

        prompt_gen_group.setLayout(prompt_gen_layout)
        layout.addWidget(prompt_gen_group)
        layout.addStretch()
        self.prompt_generation_tab.setLayout(layout)

    def setup_generate_tab(self):
        layout = QVBoxLayout()

        # Output directory
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()

        self.gen_output_dir = QLineEdit()
        output_layout.addRow("Output Directory:", self.gen_output_dir)

        output_group.setLayout(output_layout)

        # Generate button
        generate_button = QPushButton("Generate Images")
        generate_button.clicked.connect(self.start_image_generation)

        # Add to layout
        layout.addWidget(output_group)
        layout.addWidget(generate_button)
        layout.addStretch()

        self.generate_tab.setLayout(layout)

    def setup_interpolate_tab(self):  # This method follows directly
        layout = QVBoxLayout()

        # Input directory
        input_group = QGroupBox("Input Settings")
        input_layout = QFormLayout()

        self.interp_input_dir = QLineEdit()
        input_browse = QPushButton("Browse")
        input_browse.clicked.connect(self.select_interp_input_dir)

        input_dir_layout = QHBoxLayout()
        input_dir_layout.addWidget(self.interp_input_dir, 1)
        input_dir_layout.addWidget(input_browse)

        input_layout.addRow("Input Frames Directory:", input_dir_layout)
        input_group.setLayout(input_layout)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()

        self.interp_output_dir = QLineEdit()
        output_layout.addRow("Output Directory:", self.interp_output_dir)

        self.interp_fps = QSpinBox()
        self.interp_fps.setRange(15, 120)
        self.interp_fps.setValue(30)
        output_layout.addRow("Target FPS:", self.interp_fps)

        self.interp_method = QComboBox()
        self.interp_method.addItems(["OpenCV", "RIFE"])
        self.interp_method.setCurrentText("RIFE")
        output_layout.addRow("Interpolation Method:", self.interp_method)

        output_group.setLayout(output_layout)

        # Interpolate button
        interpolate_button = QPushButton("Interpolate Frames")
        interpolate_button.clicked.connect(self.start_interpolation)

        # Add to layout
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addWidget(interpolate_button)
        layout.addStretch()

        self.interpolate_tab.setLayout(layout)

    def setup_video_tab(self):
        layout = QVBoxLayout()

        # Input directory
        input_group = QGroupBox("Input Settings")
        input_layout = QFormLayout()

        self.video_frames_dir = QLineEdit()
        frames_browse = QPushButton("Browse")
        frames_browse.clicked.connect(self.select_video_frames_dir)

        frames_dir_layout = QHBoxLayout()
        frames_dir_layout.addWidget(self.video_frames_dir, 1)
        frames_dir_layout.addWidget(frames_browse)

        input_layout.addRow("Frames Directory:", frames_dir_layout)

        self.video_audio_file = QLineEdit()
        audio_browse = QPushButton("Browse")
        audio_browse.clicked.connect(self.select_video_audio_file)

        audio_file_layout = QHBoxLayout()
        audio_file_layout.addWidget(self.video_audio_file, 1)
        audio_file_layout.addWidget(audio_browse)

        input_layout.addRow("Audio File:", audio_file_layout)

        self.video_intro_file = QLineEdit()
        intro_browse = QPushButton("Browse")
        intro_browse.clicked.connect(self.select_video_intro_file)

        intro_file_layout = QHBoxLayout()
        intro_file_layout.addWidget(self.video_intro_file, 1)
        intro_file_layout.addWidget(intro_browse)

        input_layout.addRow("Intro Video (optional):", intro_file_layout)

        input_group.setLayout(input_layout)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()

        self.video_output_file = QLineEdit()
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self.select_video_output_file)

        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(self.video_output_file, 1)
        output_file_layout.addWidget(output_browse)

        output_layout.addRow("Output Video:", output_file_layout)

        self.video_fps = QSpinBox()
        self.video_fps.setRange(15, 120)
        self.video_fps.setValue(30)
        output_layout.addRow("FPS:", self.video_fps)

        self.video_resolution = QComboBox()
        self.video_resolution.addItems(["720p", "1080p", "4K"])
        self.video_resolution.setCurrentText("1080p")
        output_layout.addRow("Resolution:", self.video_resolution)

        self.video_crf = QSpinBox()
        self.video_crf.setRange(0, 51)
        self.video_crf.setValue(18)
        output_layout.addRow("Quality (CRF):", self.video_crf)

        self.video_preset = QComboBox()
        self.video_preset.addItems(
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"])
        self.video_preset.setCurrentText("slow")
        output_layout.addRow("Encoding Preset:", self.video_preset)

        output_group.setLayout(output_layout)

        # Create video button
        create_video_button = QPushButton("Create Video")
        create_video_button.clicked.connect(self.start_video_creation)

        # Add to layout
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addWidget(create_video_button)
        layout.addStretch()

        self.video_tab.setLayout(layout)

    def setup_comfyui_tab(self):
        layout = QVBoxLayout()

        # Create web view
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl(f"http://127.0.0.1:8188"))

        # Add refresh button
        refresh_button = QPushButton("Refresh ComfyUI Interface")
        refresh_button.clicked.connect(self.refresh_comfyui_interface)

        # Add to layout
        layout.addWidget(refresh_button)
        layout.addWidget(self.web_view)

        self.comfyui_tab.setLayout(layout)

    def select_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_dir = dir_path
            self.project_path_label.setText(dir_path)

            # Update other paths based on project directory
            self.gen_output_dir.setText(os.path.join(dir_path, "generated_frames"))
            self.interp_input_dir.setText(os.path.join(dir_path, "generated_frames"))
            self.interp_output_dir.setText(os.path.join(dir_path, "interpolated_frames"))
            self.video_frames_dir.setText(os.path.join(dir_path, "interpolated_frames"))
            self.video_output_file.setText(os.path.join(dir_path, "final_video.mp4"))

            # Set default ComfyUI and RIFE paths
            if not self.comfy_models_dir.text():
                self.comfy_models_dir.setText(os.path.join(dir_path, "models"))

            if not self.comfy_output_dir.text():
                self.comfy_output_dir.setText(os.path.join(dir_path, "comfyui_output"))

            if not self.rife_models_dir.text():
                self.rife_models_dir.setText(os.path.join(dir_path, "models", "rife"))

            # Look for common files in the directory
            self.auto_detect_files(dir_path)

    def auto_detect_files(self, directory):
        """Auto-detect common files in the project directory"""
        # Look for workflow file
        workflow_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if workflow_files:
            self.workflow_file = os.path.join(directory, workflow_files[0])
            self.workflow_path_label.setText(workflow_files[0])

        # Look for prompts file
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if csv_files:
            self.prompts_file = os.path.join(directory, csv_files[0])
            self.prompts_path_label.setText(csv_files[0])

        # Look for audio file
        audio_files = [f for f in os.listdir(directory) if f.endswith(('.mp3', '.wav'))]
        if audio_files:
            self.audio_file = os.path.join(directory, audio_files[0])
            self.audio_path_label.setText(audio_files[0])
            self.video_audio_file.setText(self.audio_file)

    def select_workflow_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select ComfyUI Workflow", self.project_dir,
                                                   "JSON Files (*.json)")
        if file_path:
            self.workflow_file = file_path
            self.workflow_path_label.setText(os.path.basename(file_path))

    def select_prompts_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Prompts CSV", self.project_dir, "CSV Files (*.csv)")
        if file_path:
            self.prompts_file = file_path
            self.prompts_path_label.setText(os.path.basename(file_path))

    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", self.project_dir,
                                                   "Audio Files (*.mp3 *.wav)")
        if file_path:
            self.audio_file = file_path
            self.audio_path_label.setText(os.path.basename(file_path))
            self.video_audio_file.setText(file_path)

    def select_intro_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Intro Video", self.project_dir,
                                                   "Video Files (*.mp4 *.mov)")
        if file_path:
            self.intro_file = file_path
            self.intro_path_label.setText(os.path.basename(file_path))
            self.video_intro_file.setText(file_path)

    def select_models_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Models Directory")
        if dir_path:
            self.comfy_models_dir.setText(dir_path)

    def select_comfy_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select ComfyUI Output Directory")
        if dir_path:
            self.comfy_output_dir.setText(dir_path)

    def select_rife_models_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select RIFE Models Directory")
        if dir_path:
            self.rife_models_dir.setText(dir_path)

    def select_ffmpeg_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select FFmpeg Directory")
        if dir_path:
            self.ffmpeg_path.setText(dir_path)

    def select_whisper_exe_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Whisper.cpp Executable", "", "Executables (*.exe)")
        if file_path:
            self.whisper_exe_path.setText(file_path)

    def select_whisper_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Whisper Model (GGML)", "", "Model Files (*.bin)")
        if file_path:
            self.whisper_model_path.setText(file_path)

    def select_llama_exe_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Llama.cpp Executable", "", "Executables (*.exe)")
        if file_path:
            self.llama_exe_path.setText(file_path)

    def select_llama_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Llama Model (GGUF)", "", "Model Files (*.gguf)")
        if file_path:
            self.llama_model_path.setText(file_path)

    def select_interp_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Frames Directory")
        if dir_path:
            self.interp_input_dir.setText(dir_path)

    def select_video_frames_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Frames Directory")
        if dir_path:
            self.video_frames_dir.setText(dir_path)

    def select_video_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav)")
        if file_path:
            self.video_audio_file.setText(file_path)

    def select_video_intro_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Intro Video", "", "Video Files (*.mp4 *.mov)")
        if file_path:
            self.video_intro_file.setText(file_path)

    def select_video_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output Video File", "", "Video Files (*.mp4)")
        if file_path:
            self.video_output_file.setText(file_path)

    def initialize_comfyui(self):
        models_dir = self.comfy_models_dir.text()
        output_dir = self.comfy_output_dir.text()

        if not models_dir or not os.path.exists(models_dir):
            self.show_error("Please select a valid models directory")
            return

        if not output_dir:
            output_dir = os.path.join(self.project_dir, "comfyui_output")
            os.makedirs(output_dir, exist_ok=True)
            self.comfy_output_dir.setText(output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Disable the button
        self.comfy_init_button.setEnabled(False)
        self.comfy_init_button.setText("Initializing...")

        # Initialize in a separate thread to avoid blocking the UI
        def init_thread():
            try:
                self.comfy = EmbeddedComfyUI(models_dir, output_dir)
                success = self.comfy.initialize()

                # Start the server
                if success:
                    server_url = self.comfy.start_server()

                    # Update UI from the main thread
                    self.comfy_server_input.setText(server_url.replace("http://", ""))

                    # Update button
                    self.comfy_init_button.setText("ComfyUI Initialized")
                    self.log_message("ComfyUI initialized successfully")

                    # Update the web view URL
                    self.refresh_comfyui_interface()
                else:
                    # Update button
                    self.comfy_init_button.setEnabled(True)
                    self.comfy_init_button.setText("Initialize ComfyUI")
                    self.log_message("Failed to initialize ComfyUI")
            except Exception as e:
                self.log_message(f"Error initializing ComfyUI: {str(e)}")
                self.comfy_init_button.setEnabled(True)
                self.comfy_init_button.setText("Initialize ComfyUI")

        threading.Thread(target=init_thread, daemon=True).start()

    def initialize_rife(self):
        models_dir = self.rife_models_dir.text()

        if not models_dir or not os.path.exists(models_dir):
            self.show_error("Please select a valid RIFE models directory")
            return

        # Disable the button
        self.rife_init_button.setEnabled(False)
        self.rife_init_button.setText("Initializing...")

        # Initialize in a separate thread to avoid blocking the UI
        def init_thread():
            try:
                self.rife = EmbeddedRIFE(models_dir)
                success = self.rife.initialize()

                if success:
                    # Update button
                    self.rife_init_button.setText("RIFE Initialized")
                    self.log_message("RIFE initialized successfully")
                else:
                    # Update button
                    self.rife_init_button.setEnabled(True)
                    self.rife_init_button.setText("Initialize RIFE")
                    self.log_message("Failed to initialize RIFE")
            except Exception as e:
                self.log_message(f"Error initializing RIFE: {str(e)}")
                self.rife_init_button.setEnabled(True)
                self.rife_init_button.setText("Initialize RIFE")

        threading.Thread(target=init_thread, daemon=True).start()

    def initialize_whisper(self):  # New initializer
        exe_path = self.whisper_exe_path.text()
        model_path = self.whisper_model_path.text()

        if not exe_path or not os.path.exists(exe_path):
            self.show_error("Please select a valid Whisper.cpp executable path")
            return
        if not model_path or not os.path.exists(model_path):
            self.show_error("Please select a valid Whisper model path")
            return

        self.whisper_init_button.setEnabled(False)
        self.whisper_init_button.setText("Initializing...")

        def init_thread():
            try:
                self.whisper = EmbeddedWhisper(exe_path, model_path)
                self.whisper_init_button.setText("Whisper Initialized")
                self.log_message("Whisper initialized successfully")
            except Exception as e:
                self.log_message(f"Error initializing Whisper: {str(e)}")
                self.whisper_init_button.setEnabled(True)
                self.whisper_init_button.setText("Initialize Whisper")

        threading.Thread(target=init_thread, daemon=True).start()

    def initialize_llama(self):  # New initializer
        exe_path = self.llama_exe_path.text()
        model_path = self.llama_model_path.text()

        if not exe_path or not os.path.exists(exe_path):
            self.show_error("Please select a valid Llama.cpp executable path")
            return
        if not model_path or not os.path.exists(model_path):
            self.show_error("Please select a valid Llama model path")
            return

        self.llama_init_button.setEnabled(False)
        self.llama_init_button.setText("Initializing...")

        def init_thread():
            try:
                self.llama = EmbeddedLlama(exe_path, model_path)
                self.llama_init_button.setText("Llama Initialized")
                self.log_message("Llama initialized successfully")
            except Exception as e:
                self.log_message(f"Error initializing Llama: {str(e)}")
                self.llama_init_button.setEnabled(True)
                self.llama_init_button.setText("Initialize Llama")

        threading.Thread(target=init_thread, daemon=True).start()

    def initialize_ffmpeg(self):
        ffmpeg_path = self.ffmpeg_path.text()

        if not ffmpeg_path or not os.path.exists(ffmpeg_path):
            self.show_error("Please select a valid FFmpeg directory")
            return

        # Disable the button
        self.ffmpeg_init_button.setEnabled(False)
        self.ffmpeg_init_button.setText("Initializing...")

        try:
            self.ffmpeg = EmbeddedFFmpeg(ffmpeg_path)
            self.ffmpeg_init_button.setText("FFmpeg Initialized")
            self.log_message("FFmpeg initialized successfully")
        except Exception as e:
            self.log_message(f"Error initializing FFmpeg: {str(e)}")
            self.ffmpeg_init_button.setEnabled(True)
            self.ffmpeg_init_button.setText("Initialize FFmpeg")

    def refresh_comfyui_interface(self):
        server_address = self.comfy_server_input.text()
        if not server_address.startswith("http://"):
            server_address = f"http://{server_address}"
        self.web_view.setUrl(QUrl(server_address))

    def start_prompt_generation(self):  # New start method for prompt generation
        if not self.audio_file or not os.path.exists(self.audio_file):
            self.show_error("Please select an audio file first.")
            return
        if not self.whisper:
            self.show_error("Whisper is not initialized. Please initialize it in Project Setup.")
            return
        if not self.llama:
            self.show_error("Llama is not initialized. Please initialize it in Project Setup.")
            return
        if not self.song_description_input.text():
            self.show_error("Please provide a song description for the LLM.")
            return

        self.disable_ui()
        self.console_output.clear()
        self.log_message("Starting prompt generation from audio...")

        self.worker = WorkerThread("generate_llm_prompts", {
            "audio_file": self.audio_file,
            "song_description": self.song_description_input.text(),
            "whisper": self.whisper,
            "llama": self.llama
        })
        self.worker.update_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.complete_signal.connect(self.prompt_generation_complete)
        self.worker.start()

    def prompt_generation_complete(self, result):  # New completion method
        if result.get("success"):
            self.log_message("Prompt generation complete!")
            generated_prompts_csv = result.get("output_file")
            self.prompts_file = generated_prompts_csv  # Set the generated prompts as the active prompts file
            self.prompts_path_label.setText(os.path.basename(generated_prompts_csv))

            # Display a preview of generated prompts
            preview_text = ""
            try:
                with open(generated_prompts_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        preview_text += f"[{row['Start']}-{row['End']}] {row['Prompt']}\n"
                        if i >= 10:  # Show first 10 for preview
                            preview_text += "...\n"
                            break
            except Exception as e:
                preview_text = f"Error reading generated prompts file: {e}"
            self.generated_prompts_output.setText(preview_text)
        else:
            self.log_message(f"Prompt generation failed: {result.get('error', 'Unknown error')}")
        self.enable_ui()

    def start_image_generation(self):
        """Start generating images using ComfyUI"""
        if not self.workflow_file or not os.path.exists(self.workflow_file):
            self.show_error("Workflow file not found")
            return

        if not self.prompts_file or not os.path.exists(self.prompts_file):
            self.show_error("Prompts CSV file not found")
            return

        output_dir = self.gen_output_dir.text()
        if not output_dir:
            self.show_error("Please specify an output directory")
            return

        # Check if ComfyUI is initialized
        if not self.comfy:
            response = QMessageBox.question(
                self,
                "ComfyUI Not Initialized",
                "ComfyUI is not initialized. Initialize it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if response == QMessageBox.Yes:
                self.initialize_comfyui()
                QMessageBox.information(
                    self,
                    "Initializing ComfyUI",
                    "Please wait for ComfyUI to initialize, then try again."
                )
                return
            else:
                return

        # Disable UI during processing
        self.disable_ui()

        # Clear console
        self.console_output.clear()
        self.log_message("Starting image generation...")

        # Start worker thread
        self.worker = WorkerThread("generate_images", {
            "workflow_file": self.workflow_file,
            "prompts_file": self.prompts_file,
            "output_dir": output_dir,
            "comfy": self.comfy  # Pass the ComfyUI instance
        })
        self.worker.update_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.complete_signal.connect(self.image_generation_complete)
        self.worker.start()

    def image_generation_complete(self, result):
        """Handle completion of image generation"""
        if result.get("success"):
            self.log_message(f"Image generation complete! {result.get('frame_count', 0)} frames generated.")
            self.images_dir = result.get("output_dir")

            # Update interpolation input directory
            self.interp_input_dir.setText(self.images_dir)
        else:
            self.log_message(f"Image generation failed: {result.get('error', 'Unknown error')}")

        # Re-enable UI
        self.enable_ui()

    def start_interpolation(self):
        """Start frame interpolation"""
        input_dir = self.interp_input_dir.text()
        if not input_dir or not os.path.exists(input_dir):
            self.show_error("Input frames directory not found")
            return

        output_dir = self.interp_output_dir.text()
        if not output_dir:
            self.show_error("Please specify an output directory")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Check if RIFE is initialized when using RIFE method
        if self.interp_method.currentText() == "RIFE" and not self.rife:
            response = QMessageBox.question(
                self,
                "RIFE Not Initialized",
                "RIFE is not initialized. Initialize it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if response == QMessageBox.Yes:
                self.initialize_rife()
                QMessageBox.information(
                    self,
                    "Initializing RIFE",
                    "Please wait for RIFE to initialize, then try again."
                )
                return
            else:
                return

        # Disable UI during processing
        self.disable_ui()

        # Clear console
        self.console_output.clear()
        self.log_message("Starting frame interpolation...")

        # Create temp directory
        temp_dir = os.path.join(os.path.dirname(output_dir), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Start worker thread
        self.worker = WorkerThread("interpolate_frames", {
            "input_dir": input_dir,
            "temp_dir": temp_dir,
            "output_dir": output_dir,
            "fps": self.interp_fps.value(),
            "method": self.interp_method.currentText(),
            "rife": self.rife  # Pass the RIFE instance
        })
        self.worker.update_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.complete_signal.connect(self.interpolation_complete)
        self.worker.start()

    def interpolation_complete(self, result):
        """Handle completion of frame interpolation"""
        if result.get("success"):
            self.log_message(f"Frame interpolation complete! {result.get('frame_count', 0)} frames generated.")
            self.interpolated_dir = result.get("output_dir")

            # Update video frames directory
            self.video_frames_dir.setText(self.interpolated_dir)
        else:
            self.log_message(f"Frame interpolation failed: {result.get('error', 'Unknown error')}")

        # Re-enable UI
        self.enable_ui()

    def start_video_creation(self):
        """Start video creation"""
        frames_dir = self.video_frames_dir.text()
        if not frames_dir or not os.path.exists(frames_dir):
            self.show_error("Frames directory not found")
            return

        output_file = self.video_output_file.text()
        if not output_file:
            self.show_error("Please specify an output video file")
            return

        # Check if FFmpeg is initialized
        if not self.ffmpeg:
            response = QMessageBox.question(
                self,
                "FFmpeg Not Initialized",
                "FFmpeg is not initialized. Initialize it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if response == QMessageBox.Yes:
                self.initialize_ffmpeg()
                if not self.ffmpeg:
                    return
            else:
                return

        # Disable UI during processing
        self.disable_ui()

        # Clear console
        self.console_output.clear()
        self.log_message("Starting video creation...")

        # Determine resolution
        resolution_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4K": (3840, 2160)
        }
        resolution = resolution_map.get(self.video_resolution.currentText(), (1920, 1080))

        # Start worker thread
        self.worker = WorkerThread("create_video", {
            "frames_dir": frames_dir,
            "audio_file": self.video_audio_file.text(),
            "intro_file": self.video_intro_file.text(),
            "output_file": output_file,
            "fps": self.video_fps.value(),
            "resolution": resolution,
            "crf": self.video_crf.value(),
            "preset": self.video_preset.currentText(),
            "ffmpeg": self.ffmpeg  # Pass the FFmpeg instance
        })
        self.worker.update_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.complete_signal.connect(self.video_creation_complete)
        self.worker.start()

    def video_creation_complete(self, result):
        """Handle completion of video creation"""
        if result.get("success"):
            self.log_message(f"Video creation complete!")
            self.output_video = result.get("output_file")

            # Show success message with option to open video
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText("Video created successfully!")
            msg_box.setInformativeText(f"Video saved to: {self.output_video}")
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Open)
            msg_box.setDefaultButton(QMessageBox.Ok)

            if msg_box.exec_() == QMessageBox.Open:
                # Open the video with default application
                import subprocess
                import platform

                if platform.system() == 'Windows':
                    os.startfile(self.output_video)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(('open', self.output_video))
                else:  # Linux
                    subprocess.call(('xdg-open', self.output_video))
        else:
            self.log_message(f"Video creation failed: {result.get('error', 'Unknown error')}")

        # Re-enable UI
        self.enable_ui()

    def log_message(self, message):
        """Add message to console output"""
        self.console_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def disable_ui(self):
        """Disable UI elements during processing"""
        self.tabs.setEnabled(False)
        self.progress_bar.setValue(0)

    def enable_ui(self):
        """Re-enable UI elements after processing"""
        self.tabs.setEnabled(True)

    def show_error(self, message):
        """Show error message box"""
        QMessageBox.critical(self, "Error", message)


def save_settings(self):
    """Save current settings"""
    # Project directories
    self.settings.setValue("project_dir", self.project_dir)
    self.settings.setValue("workflow_file", self.workflow_file)
    self.settings.setValue("prompts_file", self.prompts_file)
    self.settings.setValue("audio_file", self.audio_file)
    self.settings.setValue("intro_file", self.intro_file)

    # ComfyUI settings
    self.settings.setValue("comfy_models_dir", self.comfy_models_dir.text())
    self.settings.setValue("comfy_output_dir", self.comfy_output_dir.text())
    self.settings.setValue("comfy_server_address", self.comfy_server_input.text())
    self.settings.setValue("comfy_initialized", self.comfy is not None)

    # RIFE settings
    self.settings.setValue("rife_models_dir", self.rife_models_dir.text())
    self.settings.setValue("rife_initialized", self.rife is not None)

    # FFmpeg settings
    self.settings.setValue("ffmpeg_path", self.ffmpeg_path.text())
    self.settings.setValue("ffmpeg_initialized", self.ffmpeg is not None)

    # Whisper settings
    self.settings.setValue("whisper_exe_path", self.whisper_exe_path.text())
    self.settings.setValue("whisper_model_path", self.whisper_model_path.text())
    self.settings.setValue("whisper_initialized", self.whisper is not None)

    # Llama settings
    self.settings.setValue("llama_exe_path", self.llama_exe_path.text())
    self.settings.setValue("llama_model_path", self.llama_model_path.text())
    self.settings.setValue("llama_initialized", self.llama is not None)

    # Other settings
    self.settings.setValue("song_description", self.song_description_input.text())
    self.settings.setValue("current_tab", self.tabs.currentIndex())
