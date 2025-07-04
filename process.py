from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LAParams
import re
import os
import datetime
import textwrap
import argparse
import torch
from TTS.api import TTS
import glob
import logging
import sys
from typing import Optional

# Set up logger
logger = logging.getLogger("pdf_to_audio")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Allow safe loading of this config


def clean_text(text: str) -> str:
    """Normalize spaces and fix French spacing issues in the given text."""
    # Normalize spaces and fix French spacing issues
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(" .", ".").replace(" ,", ",")
    text = text.replace(" ’", "’").replace(" :", ":")
    return text.strip()

def extract_text_from_pdf_robust(path: str, num_pages: Optional[int] = None) -> str:
    """Extracts and cleans text from a PDF file, page by page."""
    laparams = LAParams(line_margin=0.3)  # Controls paragraph merging
    full_text = []
    for page_num, page_layout in enumerate(extract_pages(path, laparams=laparams), start=1):
        if num_pages is not None and page_num > num_pages:
            break
        page_lines = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_text = text_line.get_text()
                    line_text = clean_text(line_text)
                    if line_text:
                        page_lines.append(line_text)
        if page_lines:
            full_text.append(f"=== Page {page_num} ===\n" + "\n".join(page_lines))
    return "\n\n".join(full_text)

def chunk_text(text: str, max_length: int = 200000) -> list[str]:
    """Splits text into chunks of up to max_length characters, preserving paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_output_dir() -> str:
    """Return the fixed output directory name 'output'."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert PDF to audio using TTS.")
    parser.add_argument('--pdf', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--model', type=str, default="tts_models/fr/css10/vits", help='TTS model to use')
    parser.add_argument('--num-pages', type=int, default=None, help='Number of pages/chunks to transcribe in this run (resumes from last processed)')
    parser.add_argument('--language', type=str, default=None, help='Language for TTS synthesis (e.g., fr-fr)')
    parser.add_argument('--speaker', type=str, default=None, help='Speaker for TTS synthesis (if supported by model)')
    parser.add_argument('--speaker-wav', type=str, default=None, help='Path to the speaker WAV file')
    parser.add_argument('--log-level', type=str, default='info', help='Logging level (debug, info, warning, error)')
    return parser.parse_args()

def synthesize_chunks_to_audio(
    chunks: list[str],
    output_prefix: str = "page",
    model: str = "tts_models/fr/css10/vits",
    speaker: Optional[str] = None,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None
) -> None:
    """Synthesize each text chunk to an audio file using the specified TTS model and parameters."""
    logger.debug(f"synthesize_chunks_to_audio called with {len(chunks)} chunks, model={model}, speaker={speaker}, speaker_wav={speaker_wav}, language={language}")
    tts = TTS(model)
    if torch.cuda.is_available():
        tts.to("cuda")
        logger.debug("Using CUDA for TTS synthesis.")
    else:
        logger.debug("Using CPU for TTS synthesis.")

    output_dir = get_output_dir()
    logger.debug(f"Output directory for audio: {output_dir}")
    for idx, chunk in enumerate(chunks):
        file_path = os.path.join(output_dir, f"{output_prefix}_{idx+1:03}.wav")
        logger.info(f"Synthesizing chunk {idx+1}/{len(chunks)} → {file_path}")
        logger.debug(f"Chunk text (first 100 chars): {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
        tts_kwargs = {"text": chunk, "file_path": file_path}
        if speaker is not None:
            tts_kwargs["speaker"] = speaker
        if speaker_wav is not None:
            tts_kwargs["speaker_wav"] = speaker_wav
        if language is not None:
            tts_kwargs["language"] = language
        logger.debug(f"tts_to_file kwargs: {tts_kwargs}")
        tts.tts_to_file(**tts_kwargs)
        logger.debug(f"Finished synthesizing chunk {idx+1}")

#examples :
# python process.py --pdf /media/msist/data/La_parole_est_une_force.pdf --model tts_models/fr/css10/vits --num-pages 5
def main():
    args = parse_args()
    log_level = args.log_level.upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.debug(f"Log level set to {log_level}")
    logger.debug(f"Arguments received: {args}")

    pdf_path = args.pdf
    model = args.model
    num_pages = args.num_pages
    language = args.language
    speaker = args.speaker
    speaker_wav = args.speaker_wav

    output_dir = get_output_dir()
    logger.debug(f"Output directory: {output_dir}")

    logger.debug(f"Extracting text from PDF: {pdf_path}")
    raw_text = extract_text_from_pdf_robust(pdf_path, num_pages)
    logger.debug(f"Extracted text length: {len(raw_text)} characters")
    chunks = chunk_text(raw_text, max_length=1800)
    logger.debug(f"Total chunks created: {len(chunks)}")

    if not chunks:
        logger.info("No text chunks to process.")
        return
    logger.info(f"Processing {len(chunks)} chunks (pages)")
    for i, chunk in enumerate(chunks, start=1):
        logger.debug(f"Chunk {i}: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
    synthesize_chunks_to_audio(
        chunks,
        model=model,
        speaker=speaker,
        speaker_wav=speaker_wav,
        language=language,
        output_prefix="page"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
