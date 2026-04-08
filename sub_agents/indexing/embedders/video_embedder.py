from __future__ import annotations
from google.cloud import storage
import logging
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import requests
logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    logger.warning("opencv-python not installed")

from ....config import config
from .clip_embedder import embed_image_pil


def _download_video(source: str) -> str:
    if source.startswith("gs://"):
        bucket_name, blob_path = source[5:].split("/", 1)
        suffix = Path(blob_path).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        storage.Client().bucket(bucket_name).blob(blob_path).download_to_filename(tmp_path)

        return tmp_path
    
    elif source.startswith("http://") or source.startswith("https://"):

        suffix = Path(source.split("?")[0]).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        resp = requests.get(source, timeout=120, stream=True)
        resp.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return tmp_path
    else:
        return source

def _sample_frames(local_path: str, n_frames: int) -> list[Image.Image]:
    if not _CV2_OK:
        return []
    cap = cv2.VideoCapture(local_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # OpenCV uses BGR — goota convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()
    return frames

def embed_video(source: str) -> np.ndarray:
    if not _CV2_OK:
        logger.warning("OpenCV unavailable, cannot embed video")
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

    tmp_path = None
    try:
        local_path = _download_video(source)
        tmp_path = local_path if not os.path.exists(source) else None

        frames = _sample_frames(local_path, config.VIDEO_FRAMES_PER_CLIP)
        if not frames:
            logger.warning("No frames extracted from %s", source)
            return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

        embeddings = [embed_image_pil(frame) for frame in frames]
        mean_vec = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / norm) if norm > 0 else mean_vec

    except Exception as exc:
        logger.error("Video embedding failed for %s: %s", source, exc)
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
