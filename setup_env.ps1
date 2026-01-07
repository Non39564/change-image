Write-Host "Checking for existing .venv..."
if (Test-Path ".venv") {
    Write-Host "Removing old .venv..."
    Remove-Item ".venv" -Recurse -Force
}

Write-Host "Creating new .venv with Python 3.11..."
py -3.11 -m venv .venv

if (-not (Test-Path ".venv")) {
    Write-Host "Error: Failed to create .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing dependencies..."
.\.venv\Scripts\pip install -r requirements.txt

Write-Host "Done! Please restart your terminal or activating the environment using:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
