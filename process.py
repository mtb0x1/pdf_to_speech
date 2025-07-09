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
from typing import Optional, List

# Set up logger
logger = logging.getLogger("pdf_to_audio")
default_log_level = logging.INFO
logger.setLevel(default_log_level)
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

def extract_text_from_pdf_robust_per_page(path: str, num_pages: Optional[int] = None) -> List[str]:
    """Extracts and cleans text from a PDF file, returning a list of per-page texts."""
    laparams = LAParams(line_margin=0.3)
    page_texts = []
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
            page_texts.append("\n".join(page_lines))
        else:
            page_texts.append("")
    return page_texts

def chunk_text(text: str, max_length: int = 200000) -> List[str]:
    """Splits text into chunks of up to max_length characters, preserving paragraph boundaries when possible."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If the paragraph itself is longer than max_length, we need to split it
        if len(para) >= max_length:
            # First add any existing content as a chunk if we have some
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split the long paragraph into smaller pieces
            for i in range(0, len(para), max_length):
                para_piece = para[i:i+max_length]
                if i + max_length >= len(para):  # Last piece, add to current_chunk
                    current_chunk = para_piece + "\n\n"
                else:  # Complete piece, add directly to chunks
                    chunks.append(para_piece)
        # Normal case - paragraph fits within limit
        elif len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n\n"
        # Current chunk would exceed limit, start a new one
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    # Add the final chunk if there's anything left
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

def synthesize_pages_to_audio(
    pages: List[str],
    output_prefix: str = "page",
    model: str = "tts_models/fr/css10/vits",
    speaker: Optional[str] = None,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None,
    chunk_length: int = 1800
) -> None:
    """Synthesize each page's text (chunked if needed) to audio files using the specified TTS model and parameters."""
    logger.debug(f"synthesize_pages_to_audio called with {len(pages)} pages, model={model}, speaker={speaker}, speaker_wav={speaker_wav}, language={language}")
    tts = TTS(model)
    if torch.cuda.is_available():
        tts.to("cuda")
        logger.debug("Using CUDA for TTS synthesis.")
    else:
        logger.debug("Using CPU for TTS synthesis.")

    output_dir = get_output_dir()
    logger.debug(f"Output directory for audio: {output_dir}")
    for page_idx, page_text in enumerate(pages):
        page_number = page_idx + 1
        chunks = chunk_text(page_text, max_length=chunk_length)
        if not chunks:
            logger.info(f"No text to process for page {page_number}.")
            continue
        for chunk_idx, chunk in enumerate(chunks):
            chunk_number = chunk_idx + 1
            file_path = os.path.join(output_dir, f"{output_prefix}_{page_number:03}_part_{chunk_number}.wav")
            logger.info(f"Synthesizing page {page_number} part {chunk_number}/{len(chunks)} → {file_path}")
            logger.debug(f"Chunk text (first 100 chars): {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
            tts_kwargs = {"text": chunk, "file_path": file_path}
            if speaker is not None:
                tts_kwargs["speaker"] = speaker
            if speaker_wav is not None:
                tts_kwargs["speaker_wav"] = speaker_wav
            if language is not None:
                tts_kwargs["language"] = language
            logger.debug(f"tts_to_file kwargs: {tts_kwargs}")
            try:
                tts.tts_to_file(**tts_kwargs)
                logger.debug(f"Finished synthesizing page {page_number} part {chunk_number}")
            except Exception as e:
                logger.error(f"Failed to synthesize page {page_number} part {chunk_number}: {e}")
                logger.debug(f"Continuing with next chunk...")

#examples :
# python process.py --pdf /media/msist/data/La_parole_est_une_force.pdf --model tts_models/fr/css10/vits --num-pages 5
def main():
    args = parse_args()
    if args.log_level is not None :
        logger.setLevel(getattr(logging, args.log_level.upper() , default_log_level))
    logger.info(f"Log level set to {logging.getLevelName(logger.level)}")
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
    page_texts = extract_text_from_pdf_robust_per_page(pdf_path, num_pages)
    logger.debug(f"Extracted {len(page_texts)} pages of text")

    if not page_texts:
        logger.info("No text pages to process.")
        return
    logger.info(f"Processing {len(page_texts)} pages")
    for i, page in enumerate(page_texts, start=1):
        logger.debug(f"Page {i} text (first 100 chars): {page[:100]}{'...' if len(page) > 100 else ''}")
    synthesize_pages_to_audio(
        page_texts,
        model=model,
        speaker=speaker,
        speaker_wav=speaker_wav,
        language=language,
        output_prefix="page",
        chunk_length=1800
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
