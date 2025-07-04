# PDF to Speech (PTS)

This project converts PDF documents into speech audio files using a Text-to-Speech (TTS) model. It extracts text from a PDF, splits it into manageable chunks, and synthesizes each chunk into an audio file using the [mute.sh by Kyutai](https://github.com/kyutai-labs/delayed-streams-modeling) TTS system.

## Features
- Extracts and cleans text from PDF files (page by page)
- Supports chunking of text for long documents
- Synthesizes audio using customizable TTS models (including language and speaker options)
- Outputs audio files (WAV) for each chunk/page

## Requirements
- Python 3.8+
- [moshi](https://pypi.org/project/moshi/) (Kyutai's TTS Python package)
- pdfminer.six
- torch

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mtb0x1/pdf_to_speech
   cd pdf_to_speech
   ```
2. Create a new branch for mute.sh implementation:
   ```bash
   git checkout -b mute.sh
   ```
3. Install dependencies:
   ```bash
   pip install moshi pdfminer.six torch
   ```
   
   Alternatively, if you have [uv](https://docs.astral.sh/uv/) installed:
   ```bash
   uv pip install moshi pdfminer.six torch
   ```

## Usage

Run the script from the command line:

```bash
python process.py --pdf <path_to_pdf> [--model <tts_model>] [--num-pages <N>] [--language <lang>] [--speaker <name>] [--speaker-wav <wav_path>] [--log-level <level>]
```

### Arguments
- `--pdf` (required): Path to the PDF file to convert.
- `--model`: Kyutai TTS model to use (default: `kyutai/tts`).
- `--num-pages`: Number of pages to process (default: all pages).
- `--language`: Language code for TTS synthesis (default: `fr` for French).
- `--speaker`: Speaker name (if supported by the model).
- `--speaker-wav`: Path to a reference speaker WAV file (if supported).
- `--log-level`: Logging level (`debug`, `info`, `warning`, `error`).

### Example

Convert the first 5 pages of a French PDF to audio using the Kyutai TTS model:

```bash
python process.py --pdf file.pdf --model kyutai/tts --language fr --num-pages 5
```

The resulting audio files will be saved in the `output/` directory as `page_001.wav`, `page_002.wav`, etc.

## Notes
- For best results, use a TTS model that matches the language of your PDF. French is set as the default language.
- Kyutai TTS models are optimized for real-time usage and provide high-quality speech synthesis.
- For more information about available models and features, see the [Kyutai TTS documentation](https://github.com/kyutai-labs/delayed-streams-modeling).
- If you encounter CUDA errors, ensure your system has a compatible GPU and CUDA drivers, or the script will fall back to CPU.

## License
MIT License 