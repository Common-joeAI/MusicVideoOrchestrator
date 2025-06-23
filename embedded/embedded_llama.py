import os
import subprocess
from typing import Dict, Any, Optional

class EmbeddedLlama:
    """Class to embed llama.cpp functionality for LLM inference."""

    def __init__(self, llama_exe_path: str, model_path: str):
        self.llama_exe_path = os.path.abspath(llama_exe_path)
        self.model_path = os.path.abspath(model_path)

        if not os.path.exists(self.llama_exe_path):
            raise FileNotFoundError(f"llama.cpp executable not found at {self.llama_exe_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Llama model not found at {self.model_path}")

        print(f"Llama initialized with executable: {self.llama_exe_path}")
        print(f"Llama model: {self.model_path}")

    def generate_text(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        Generates text using the llama.cpp model.
        """
        cmd = [
            self.llama_exe_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--log-disable" # Disable verbose logging from llama.cpp
        ]

        print(f"Running Llama command: {' '.join(cmd[:3])} ...") # Avoid printing full prompt
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            # llama.cpp outputs the prompt first, then the completion
            # We need to extract only the generated part
            generated_text = process.stdout.replace(prompt, "", 1).strip()
            return generated_text

        except subprocess.CalledProcessError as e:
            print(f"Llama.cpp error: {e.stderr}")
            raise
        except Exception as e:
            print(f"Error running Llama: {str(e)}")
            raise

