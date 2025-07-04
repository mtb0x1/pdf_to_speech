# PDF to Speech (PTS)

This project converts PDF documents into speech audio files using a Text-to-Speech (TTS) model. It extracts text from a PDF, splits it into manageable chunks, and synthesizes each chunk into an audio file using the Kokoro TTS model.

## Features
- Extracts and cleans text from PDF files (page by page)
- Supports chunking of text for long documents
- Synthesizes audio using customizable TTS models (including language and speaker options)
- Outputs audio files (WAV) for each chunk/page

## Requirements
- Python 3.8+
- Kokoro TTS
- pdfminer.six
- torch
- Additional dependencies as required by Kokoro

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mtb0x1/pdf_to_speech
   cd pdf_to_speech
   ```
2. Install dependencies:
   ```bash
   pip install kokoro pdfminer.six torch
   ```
   You may need to install additional dependencies for your Kokoro TTS model.
   
   <!-- TODO: Add specific installation instructions for Kokoro when available -->

## Usage

Run the script from the command line:

```bash
python process.py --pdf <path_to_pdf> [--model <tts_model>] [--num-pages <N>] [--language <lang>] [--speaker <name>] [--speaker-wav <wav_path>] [--log-level <level>]
```

### Arguments
- `--pdf` (required): Path to the PDF file to convert.
- `--model`: Kokoro TTS model to use (default: `kokoro_models/default`).
- `--num-pages`: Number of pages to process (default: all pages).
- `--language`: Language code for TTS synthesis (e.g., `fr-fr`).
- `--speaker`: Speaker name (if supported by the model).
- `--speaker-wav`: Path to a reference speaker WAV file (if supported).
- `--log-level`: Logging level (`debug`, `info`, `warning`, `error`).

### Example

Convert the first 5 pages of a French PDF to audio using the `kokoro_models/fr/default` model:

```bash
python process.py --pdf file.pdf --model kokoro_models/fr/default --num-pages 5
```

The resulting audio files will be saved in the `output/` directory as `page_001.wav`, `page_002.wav`, etc.

## Notes
- For best results, use a Kokoro TTS model that matches the language of your PDF.
- Some models support multiple speakers or require additional arguments.
- If you encounter CUDA errors, ensure your system has a compatible GPU and CUDA drivers, or the script will fall back to CPU.
- The language parameter is important for proper alignment with the language of your PDF.

<!-- TODO: Add more specific information about Kokoro TTS models and their capabilities when available -->

## License
MIT License 