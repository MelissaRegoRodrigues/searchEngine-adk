# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""ADK tool functions used by the indexing_agent.

index_new_products(brand)
  → reads BigQuery for products NOT yet in FAISS
  → for each product generates embeddings for text, images, audios, videos
  → adds them to the FAISSStore and saves to disk
  → returns a summary string for the LLM

get_index_stats()
  → returns current FAISS index stats (vector counts, paths)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from ..config import config
from ..store.faiss_store import faiss_store
from ..sub_agents.indexing.embedders.clip_embedder import embed_text, embed_image
from ..sub_agents.indexing.embedders.audio_embedder import embed_audio
from ..sub_agents.indexing.embedders.video_embedder import embed_video

# ── BigQuery helper ───────────────────────────────────────────────────────────

try:
    from google.cloud import bigquery  # type: ignore
    _BQ_OK = True
except ImportError:
    _BQ_OK = False

_MOCK_PRODUCTS: list[dict[str, Any]] = [
    {
        "product_id": "mock-001",
        "brand": "MockBrand",
        "title": "Kids' Running Shoes",
        "description": "Lightweight shoes for active kids with breathable mesh upper.",
        "attributes": json.dumps({"size": "10T", "color": "Blue"}),
        "image_urls": json.dumps([]),
        "audio_urls": json.dumps([]),
        "video_urls": json.dumps([]),
    },
    {
        "product_id": "mock-002",
        "brand": "MockBrand",
        "title": "Adult Trail Runners",
        "description": "High-grip trail running shoes with foam midsole.",
        "attributes": json.dumps({"size": "10", "color": "Black"}),
        "image_urls": json.dumps([]),
        "audio_urls": json.dumps([]),
        "video_urls": json.dumps([]),
    },
]


def _fetch_products_from_bq(brand: str) -> list[dict[str, Any]]:
    if not _BQ_OK or config.BQ_PROJECT_ID == "your-gcp-project":
        logger.info("Using mock products for brand '%s'", brand)
        return [p for p in _MOCK_PRODUCTS if p["brand"].lower() == brand.lower()] or _MOCK_PRODUCTS

    client = bigquery.Client(project=config.BQ_PROJECT_ID)
    query = f"""
        SELECT product_id, brand, title, description, attributes,
               image_urls, audio_urls, video_urls
        FROM `{config.BQ_PROJECT_ID}.{config.BQ_DATASET_ID}.{config.BQ_TABLE_ID}`
        WHERE LOWER(brand) = LOWER(@brand)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("brand", "STRING", brand)]
    )
    return [dict(row) for row in client.query(query, job_config=job_config).result()]


def _parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ── ADK tool functions ────────────────────────────────────────────────────────

def index_new_products(brand: str) -> str:
    """Index only the products for *brand* that are not yet in the FAISS store.

    For each new product the tool generates:
      • 1 text embedding  (title + description + attributes combined)
      • 1 embedding per image URL  (CLIP image encoder)
      • 1 embedding per audio URL  (Whisper → CLIP text encoder)
      • 1 embedding per video URL  (sampled frames → mean CLIP image embedding)

    All embeddings are added to the FAISSStore and flushed to disk.

    Returns a plain-text summary suitable for the LLM to relay to the user.
    """
    products = _fetch_products_from_bq(brand)
    if not products:
        return f"No products found in BigQuery for brand '{brand}'."

    new_count = 0
    skipped   = 0
    errors: list[str] = []

    for product in products:
        pid = product.get("product_id", "unknown")

        if faiss_store.already_indexed(pid):
            skipped += 1
            continue

        # ── Text embedding ────────────────────────────────────────────────────
        text_parts = [
            product.get("title", ""),
            product.get("description", ""),
            product.get("attributes", ""),
        ]
        combined_text = " | ".join(p for p in text_parts if p)
        try:
            text_vec = embed_text(combined_text)
            faiss_store.add_text_embedding(pid, text_vec, source="text_fields")
        except Exception as e:
            errors.append(f"{pid} text: {e}")

        # ── Image embeddings ─────────────────────────────────────────────────
        for url in _parse_json_list(product.get("image_urls")):
            try:
                vec = embed_image(url)
                faiss_store.add_image_embedding(pid, vec, source=url)
            except Exception as e:
                errors.append(f"{pid} image({url}): {e}")

        # ── Audio embeddings ─────────────────────────────────────────────────
        for url in _parse_json_list(product.get("audio_urls")):
            try:
                vec, transcript = embed_audio(url)
                if vec.any():
                    faiss_store.add_text_embedding(pid, vec, source=f"audio:{url}")
            except Exception as e:
                errors.append(f"{pid} audio({url}): {e}")

        # ── Video embeddings ─────────────────────────────────────────────────
        for url in _parse_json_list(product.get("video_urls")):
            try:
                vec = embed_video(url)
                if vec.any():
                    faiss_store.add_image_embedding(pid, vec, source=f"video:{url}")
            except Exception as e:
                errors.append(f"{pid} video({url}): {e}")

        new_count += 1

    faiss_store.save()

    summary_lines = [
        f"Indexing complete for brand '{brand}'.",
        f"  • Products indexed : {new_count}",
        f"  • Already indexed  : {skipped} (skipped)",
    ]
    if errors:
        summary_lines.append(f"  • Errors           : {len(errors)}")
        for err in errors[:5]:   # cap log noise
            summary_lines.append(f"    - {err}")
    summary_lines.append("")
    summary_lines.append(str(faiss_store.stats()))
    return "\n".join(summary_lines)


def get_index_stats() -> str:
    """Return current FAISS index statistics as a formatted string."""
    stats = faiss_store.stats()
    return (
        f"FAISS Index Stats\n"
        f"  Text  vectors : {stats['text_vectors']}\n"
        f"  Image vectors : {stats['image_vectors']}\n"
        f"  Metadata rows : {stats['metadata_entries']}\n"
        f"  Index dir     : {stats['index_dir']}"
    )
