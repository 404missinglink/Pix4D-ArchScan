import os
import sys
import subprocess

huggingface_bin_path = "/home/user/.local/bin"
os.environ["PATH"] = f"{huggingface_bin_path}:{os.environ['PATH']}"

subprocess.run(["huggingface-cli", "login", "--token", 'PLACEHOLDER KEY'])
