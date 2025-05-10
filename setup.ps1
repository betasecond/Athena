# Athena Project Setup Script
# This script installs dependencies and prepares the environment

Write-Host "Setting up Athena - RAG-powered AI customer service agent for logistics" -ForegroundColor Cyan

# Create a .env file if it doesn't exist
if (-not(Test-Path -Path ".env" -PathType Leaf)) {
    Write-Host "Creating .env file with sample configuration..." -ForegroundColor Yellow
    @"
# Athena Environment Configuration

# API settings
DEBUG=true

# Database
DATABASE_URL=sqlite:///./athena.db

# Vector Database
VECTOR_DB_PATH=./vector_db

# OpenAI API settings (replace with your actual API key)
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# Security (change for production)
SECRET_KEY=dev_secret_key
"@ | Set-Content -Path ".env"
    Write-Host "Created .env file. Please edit it to add your OpenAI API key." -ForegroundColor Green
}

# Install dependencies with UV, handling the onnxruntime issue
Write-Host "Installing dependencies with UV..." -ForegroundColor Cyan

try {
    # First try to install the project without onnxruntime
    uv pip install -e . --exclude onnxruntime
} catch {
    Write-Host "Falling back to alternative installation method..." -ForegroundColor Yellow
    
    # Try with a Windows-compatible alternative
    try {
        # First install main package dependencies
        uv pip install -e .
        
        # If onnxruntime is needed but fails, try a Windows-compatible version
        try {
            uv pip uninstall -y onnxruntime
            uv pip install onnxruntime-directml
        } catch {
            Write-Host "Could not install onnxruntime-directml. Your system may not need ONNX support for basic functionality." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Installation encountered issues. Please see error messages above." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Setting up development database..." -ForegroundColor Cyan
# Here you would typically run database migrations
# For now, we're using SQLite which will be created on first run

Write-Host "Setup completed!" -ForegroundColor Green
Write-Host @"

To start the development server, run:
uv pip run python -m uvicorn athena.main:app --reload --host 0.0.0.0 --port 8000

Then open your browser to http://localhost:8000/docs to see the API documentation.

Don't forget to update your .env file with your actual OpenAI API key!
"@ -ForegroundColor Cyan