from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import whisper  # type: ignore
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    logger.warning("openai-whisper not installed")

from ....config import config
from .clip_embedder import embed_text

_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", config.WHISPER_MODEL)
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)


def _download_audio(source: str) -> str:
    """Download audio to a temp file and return the local path."""
    if source.startswith("gs://"):
        from google.cloud import storage  # type: ignore
        bucket_name, blob_path = source[5:].split("/", 1)
        suffix = Path(blob_path).suffix or ".mp3"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        client = storage.Client()
        client.bucket(bucket_name).blob(blob_path).download_to_filename(tmp_path)
        return tmp_path
    
    elif source.startswith("http://") or source.startswith("https://"):
        import requests
        suffix = Path(source.split("?")[0]).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        resp = requests.get(source, timeout=60)
        resp.raise_for_status()

        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        return tmp_path
    
    else:
        return source  


def embed_audio(source: str) -> tuple[np.ndarray, str]:
    if not _WHISPER_OK:
        logger.warning("Whisper unavailable — returning zero vector")
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32), ""

    _load_whisper()
    tmp_path = None
    try:
        local_path = _download_audio(source)
        tmp_path = local_path if not os.path.exists(source) else None

        result = _whisper_model.transcribe(local_path)
        transcript: str = result.get("text", "").strip()
        logger.info("Whisper transcript (%s): %s…", source, transcript[:80])

        if not transcript:
            logger.warning("Empty transcript for %s", source)
            return np.zeros(config.EMBEDDING_DIM, dtype=np.float32), ""

        embedding = embed_text(transcript)
        return embedding, transcript

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
