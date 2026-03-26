import subprocess
import os

# Check the system CUDA compiler version
print("Checking system NVCC version:")
subprocess.run("/usr/local/cuda/bin/nvcc --version", shell=True)
print("-" * 50)

env = os.environ.copy()
env["PYTHONPATH"] = f"/kaggle/working:{env.get('PYTHONPATH', '')}"

# Force the architecture and the build process
env["TORCH_CUDA_ARCH_LIST"] = "12.0"
env["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"

# Limit parallel jobs to prevent OOM crashes
env["MAX_JOBS"] = "2"

commands = [
    "uv pip uninstall torch torchvision torchaudio",
    "uv pip install --target=/kaggle/working nvidia-cutlass",
]

for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)
