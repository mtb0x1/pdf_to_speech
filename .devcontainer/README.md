# DevContainer for PDF to Speech

This folder contains the configuration for a VS Code DevContainer, which provides a reproducible development environment with all dependencies pre-installed.

## Features
- Python 3.10
- System dependencies for audio processing (libsndfile1, ffmpeg)
- All required Python dependencies:
  - `TTS` - Text-to-Speech library
  - `pdfminer.six` - PDF text extraction
  - `torch` - Machine learning framework
- Development tools:
  - `black` - Code formatter
  - `flake8` - Linter
  - `mypy` - Static type checker
- VS Code extensions for Python development:
  - Python extension
  - Pylance (Python language server)
  - Black formatter
  - Flake8 linter
  - Auto Docstring generator

## Configuration
- Automatic code formatting on save with Black
- Flake8 linting enabled
- Basic type checking with Pylance

## Usage
1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open this project folder in VS Code.
3. When prompted, reopen in the container (or use the Command Palette: "Dev Containers: Reopen in Container").
4. The environment will be built and all dependencies installed automatically.

You can now run, debug, and develop the project in a consistent environment. The container includes everything needed to convert PDF documents to speech using the TTS library.

## System Requirements
- At least 4GB of RAM for basic usage
- 8GB+ RAM recommended for processing larger documents
- GPU with CUDA support (optional) for faster speech synthesis
- At least 2GB of free disk space