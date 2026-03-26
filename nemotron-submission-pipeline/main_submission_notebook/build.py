import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = f"/kaggle/working:{env.get('PYTHONPATH', '')}"

commands = [
    "uv pip uninstall torch torchvision torchaudio",
    "uv pip install  --system --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128",
    "uv pip install  --system --no-build-isolation 'causal-conv1d>=1.4.0'",
    "uv pip install  --system --no-build-isolation 'git+https://github.com/state-spaces/mamba.git'",
]

for cmd in commands:
    print(f"Running: {cmd}")
    # Pass the updated env to subprocess
    subprocess.run(cmd, shell=True, check=True, env=env)
