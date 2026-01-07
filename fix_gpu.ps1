Write-Host "Uninstalling CPU-only torch..."
.\.venv\Scripts\pip uninstall -y torch torchvision torchaudio

Write-Host "Installing Torch with CUDA support (Large download ~2.5GB)..."
.\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Verifying installation..."
.\.venv\Scripts\python check_gpu.py
