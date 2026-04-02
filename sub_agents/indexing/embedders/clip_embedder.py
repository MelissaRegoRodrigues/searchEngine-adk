from __future__ import annotations
import logging
from io import BytesIO
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False
    logger.warning("transformers not installed")

from ....config import config

_model = None
_processor = None

def _load_clip():
    #Load CLIP model and processor lazily to save memory if not used
    global _model, _processor
    if _model is None:
        logger.info("Loading CLIP model", config.CLIP_MODEL)
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        _model, _processor = model, processor

def _normalise(vec: np.ndarray) -> np.ndarray:
    #Transform to unit vector for cosine similarity search in FAISS
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)

def embed_text(text: str) -> np.ndarray:
    # Embed text using CLIP
    if not _CLIP_OK:
        logger.warning("CLIP unavailable — cannot embed text")
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

    _load_clip()
    inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        features = _model.get_text_features(**inputs)
    return _normalise(features[0].cpu().numpy()) #return as 1D float32 array

def _open_image(source: str | Path) -> Image.Image:
    #Open image from local path, GCS URI, or HTTP URL
    src = str(source)
    if src.startswith("gs://"):
        try:
            from google.cloud import storage  # type: ignore
            bucket_name, blob_path = src[5:].split("/", 1)
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(blob_path)
            data = blob.download_as_bytes()
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Could not fetch GCS object {src}: {e}") from e
        
    elif src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    
    else:
        return Image.open(src).convert("RGB")

def embed_image(source: str | Path) -> np.ndarray:
    # Embed an image file using CLIP
    if not _CLIP_OK:
        logger.warning("CLIP unavailable — cannot embed image")
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

    _load_clip()
    import torch
    image = _open_image(source)
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = _model.get_image_features(**inputs)
    return _normalise(features[0].cpu().numpy())


def embed_image_pil(image: Image.Image) -> np.ndarray:
    # Embed a PIL image using CLIP
    if not _CLIP_OK:
        logger.warning("CLIP unavailable — cannot embed image")
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    _load_clip()
    import torch
    inputs = _processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        features = _model.get_image_features(**inputs)
    return _normalise(features[0].cpu().numpy())
