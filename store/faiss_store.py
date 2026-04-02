# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""FAISS index manager — persists two indexes (text + image) to disk.

Layout on disk (INDEX_DIR):
  text.index      – FAISS flat inner-product index for CLIP text embeddings
  image.index     – FAISS flat inner-product index for CLIP image embeddings
  metadata.json   – maps each FAISS int-id → {product_id, modality, source}

Both indexes share the same int-id counter so a single metadata lookup works
regardless of which index returned the hit.

Vectors are L2-normalised before insertion, so inner-product == cosine similarity.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False
    logger.warning("faiss-cpu not installed — run: pip install faiss-cpu")

from ..config import config


class FAISSStore:
    """Wrapper around two FAISS flat indexes (text and image modalities)."""

    def __init__(self) -> None:
        self.index_dir = Path(config.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._text_path  = self.index_dir / config.FAISS_TEXT_INDEX
        self._image_path = self.index_dir / config.FAISS_IMAGE_INDEX
        self._meta_path  = self.index_dir / config.METADATA_FILE

        self._meta: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0

        self._text_index  = self._load_or_create(self._text_path)
        self._image_index = self._load_or_create(self._image_path)
        self._load_metadata()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_or_create(self, path: Path):
        if not _FAISS_OK:
            return None
        if path.exists():
            logger.info("Loading FAISS index from %s", path)
            return faiss.read_index(str(path))
        logger.info("Creating new FAISS index at %s", path)
        # IndexFlatIP with L2-normed vectors gives cosine similarity scores in [−1, 1]
        return faiss.IndexFlatIP(config.EMBEDDING_DIM)

    def _load_metadata(self) -> None:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                raw = json.load(f)
            self._meta = {int(k): v for k, v in raw.items()}
            self._next_id = max(self._meta.keys(), default=-1) + 1
            logger.info("Loaded %d metadata entries from disk", len(self._meta))

    def _save_metadata(self) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

    def _save_indexes(self) -> None:
        if not _FAISS_OK:
            return
        faiss.write_index(self._text_index,  str(self._text_path))
        faiss.write_index(self._image_index, str(self._image_path))

    def _add(
        self,
        index,
        product_id: str,
        embedding: np.ndarray,
        modality: str,
        source: str,
    ) -> int:
        if not _FAISS_OK:
            return -1
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        index.add(vec)
        fid = self._next_id
        self._meta[fid] = {
            "product_id": product_id,
            "modality": modality,
            "source": source,
        }
        self._next_id += 1
        return fid

    def _search(
        self, index, embedding: np.ndarray, top_k: int
    ) -> list[dict[str, Any]]:
        if not _FAISS_OK or index is None or index.ntotal == 0:
            return []
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        scores, ids = index.search(vec, min(top_k, index.ntotal))
        results = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            meta = self._meta.get(int(fid), {})
            results.append({**meta, "score": float(score), "faiss_id": int(fid)})
        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def already_indexed(self, product_id: str) -> bool:
        """Returns True if any entry for this product_id exists in metadata."""
        return any(v["product_id"] == product_id for v in self._meta.values())

    def add_text_embedding(
        self, product_id: str, embedding: np.ndarray, source: str = ""
    ) -> int:
        return self._add(self._text_index, product_id, embedding, "text", source)

    def add_image_embedding(
        self, product_id: str, embedding: np.ndarray, source: str = ""
    ) -> int:
        return self._add(self._image_index, product_id, embedding, "image", source)

    def save(self) -> None:
        """Flush both indexes and metadata JSON to disk."""
        self._save_indexes()
        self._save_metadata()
        logger.info("FAISS store saved — %d total entries", len(self._meta))

    def search_text(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Nearest-neighbour search in the text embedding space."""
        return self._search(self._text_index, query_embedding, top_k)

    def search_image(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Nearest-neighbour search in the image embedding space."""
        return self._search(self._image_index, query_embedding, top_k)

    def stats(self) -> dict[str, Any]:
        return {
            "text_vectors":  self._text_index.ntotal  if _FAISS_OK and self._text_index  else 0,
            "image_vectors": self._image_index.ntotal if _FAISS_OK and self._image_index else 0,
            "metadata_entries": len(self._meta),
            "index_dir": str(self.index_dir),
        }


# Module-level singleton shared across the whole process
faiss_store = FAISSStore()
