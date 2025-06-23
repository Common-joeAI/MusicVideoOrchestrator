import os
import sys
import threading
import time
import json
import subprocess
import requests
from typing import Dict, Any, Optional, List, Tuple


class EmbeddedComfyUI:
    """Class to embed ComfyUI functionality using its API"""

    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = os.path.abspath(models_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.server_address = "127.0.0.1"
        self.server_port = 8188
        self.server_process = None
        self.server_thread = None
        self.prompt_id_counter = 0
        self.results = {}
        self.initialized = False
        self.use_gpu = True  # Default to trying GPU first

    def initialize(self):
        """Initialize ComfyUI systems"""
        if self.initialized:
            return True

        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing ComfyUI: {str(e)}")
            return False

    def start_server(self):
        """Start the ComfyUI server in a separate process"""
        if self.is_server_running():
            return f"http://{self.server_address}:{self.server_port}"  # Server already running

        comfyui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui")

        # Prepare environment variables
        env = os.environ.copy()

        # Start ComfyUI process
        cmd = [
            sys.executable,
            os.path.join(comfyui_dir, "main.py"),
            "--port", str(self.server_port)
        ]

        # Add CPU flag if not using GPU
        if not self.use_gpu:
            cmd.append("--cpu")
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        print(f"Starting ComfyUI server with command: {' '.join(cmd)}")
        print(f"Working directory: {comfyui_dir}")
        print(f"GPU mode: {'enabled' if self.use_gpu else 'disabled'}")

        try:
            self.server_process = subprocess.Popen(
                cmd,
                cwd=comfyui_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        except Exception as e:
            print(f"Error starting ComfyUI: {str(e)}")
            # If failed with GPU, try with CPU
            if self.use_gpu:
                print("Retrying with CPU mode...")
                self.use_gpu = False
                return self.start_server()
            else:
                raise

        # Wait for server to start
        def monitor_output():
            while self.server_process and self.server_process.poll() is None:
                line = self.server_process.stdout.readline()
                if not line:
                    break
                print(f"ComfyUI: {line.strip()}")

                # Check for CUDA errors and switch to CPU if needed
                if "CUDA" in line and ("error" in line.lower() or "not compiled with CUDA" in line):
                    print("CUDA error detected. Restarting in CPU mode...")
                    self.server_process.terminate()
                    self.use_gpu = False
                    self.start_server()
                    return

                if "Starting server" in line:
                    print("ComfyUI server started!")
                    break

        self.server_thread = threading.Thread(target=monitor_output)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Wait for server to start
        max_wait = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.is_server_running():
                print("ComfyUI server is responding!")
                break
            time.sleep(1)

        return f"http://{self.server_address}:{self.server_port}"

    def is_server_running(self):
        """Check if ComfyUI server is running and responding"""
        try:
            response = requests.get(f"http://{self.server_address}:{self.server_port}/system_stats", timeout=2)
            return response.status_code == 200
        except:
            return False

    def execute_workflow(self, workflow: Dict[str, Any], prompt_id: Optional[str] = None) -> str:
        """Execute a ComfyUI workflow via API"""
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize ComfyUI")

        if not self.is_server_running():
            raise RuntimeError("ComfyUI server is not running")

        self.prompt_id_counter += 1
        prompt_id = prompt_id or f"embedded_{self.prompt_id_counter}"

        # Prepare the API request
        client_id = f"embedded_comfy_{int(time.time())}"

        data = {
            "prompt": workflow,
            "client_id": client_id
        }

        # Queue the prompt
        response = requests.post(
            f"http://{self.server_address}:{self.server_port}/prompt",
            json=data
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to queue prompt: {response.text}")

        result = response.json()
        return result.get("prompt_id", prompt_id)

    def get_execution_status(self, prompt_id: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Get the status of a workflow execution"""
        # Check if execution is complete
        if prompt_id in self.results:
            return True, 1.0, self.results[prompt_id]

        # Check the history endpoint
        try:
            response = requests.get(f"http://{self.server_address}:{self.server_port}/history/{prompt_id}")
            if response.status_code == 200:
                prompt_info = response.json()

                # Check if execution is complete
                if "outputs" in prompt_info:
                    self.results[prompt_id] = prompt_info
                    return True, 1.0, prompt_info

                # Check progress
                if "progress" in prompt_info:
                    progress = prompt_info["progress"]
                    return False, progress, {"progress": progress}
        except Exception as e:
            print(f"Error checking execution status: {e}")

        # Default: still executing
        return False, 0.0, {}

    def wait_for_execution(self, prompt_id: str, timeout: float = 300) -> Dict[str, Any]:
        """Wait for a workflow execution to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            completed, progress, result = self.get_execution_status(prompt_id)

            if completed:
                return result

            time.sleep(0.1)

        raise TimeoutError(f"Execution timed out after {timeout} seconds")

    def get_image_path(self, output_data: Dict[str, Any]) -> Optional[str]:
        """Extract image path from output data"""
        if "outputs" in output_data:
            for node_id, node_output in output_data["outputs"].items():
                if "images" in node_output:
                    for image_data in node_output["images"]:
                        filename = image_data.get("filename")
                        if filename:
                            # Download the image to our output directory
                            image_url = f"http://{self.server_address}:{self.server_port}/view?filename={filename}"
                            local_path = os.path.join(self.output_dir, os.path.basename(filename))

                            response = requests.get(image_url, stream=True)
                            if response.status_code == 200:
                                with open(local_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                return local_path
        return None

    def cleanup(self):
        """Clean up resources"""
        if self.server_process and self.server_process.poll() is None:
            try:
                print("Stopping ComfyUI server...")
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("ComfyUI server stopped")
            except:
                try:
                    print("Forcing ComfyUI server to stop...")
                    self.server_process.kill()
                    print("ComfyUI server killed")
                except:
                    print("Failed to stop ComfyUI server")
