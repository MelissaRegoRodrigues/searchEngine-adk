# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""ADK tool functions used by the search_agent.

search_by_text(query, top_k)
  → encode text with CLIP → search both FAISS indexes → merge + fetch BQ metadata

search_by_image(gcs_uri_or_url, top_k)
  → encode image with CLIP → search image FAISS index → fetch BQ metadata

search_by_audio(gcs_uri_or_url, top_k)
  → Whisper transcription → CLIP text encode → search text index

search_by_video(gcs_uri_or_url, top_k)
  → sample frames → mean CLIP image embed → search image index
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from ..config import config
from ..store.faiss_store import faiss_store
from ..sub_agents.indexing.embedders.clip_embedder import embed_text, embed_image
from ..sub_agents.indexing.embedders.audio_embedder import embed_audio
from ..sub_agents.indexing.embedders.video_embedder import embed_video

# ── BigQuery product detail lookup ────────────────────────────────────────────

try:
    from google.cloud import bigquery  # type: ignore
    _BQ_OK = True
except ImportError:
    _BQ_OK = False

_MOCK_DETAIL: dict[str, dict] = {
    "mock-001": {
        "product_id": "mock-001",
        "brand": "MockBrand",
        "title": "Kids' Running Shoes",
        "description": "Lightweight shoes for active kids.",
        "attributes": '{"size":"10T","color":"Blue"}',
    },
    "mock-002": {
        "product_id": "mock-002",
        "brand": "MockBrand",
        "title": "Adult Trail Runners",
        "description": "High-grip trail running shoes.",
        "attributes": '{"size":"10","color":"Black"}',
    },
}


def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
    """Return a {product_id: row_dict} map for the given IDs."""
    if not product_ids:
        return {}

    if not _BQ_OK or config.BQ_PROJECT_ID == "your-gcp-project":
        return {pid: _MOCK_DETAIL[pid] for pid in product_ids if pid in _MOCK_DETAIL}

    client = bigquery.Client(project=config.BQ_PROJECT_ID)
    placeholders = ", ".join(f"'{pid}'" for pid in product_ids)
    query = f"""
        SELECT product_id, brand, title, description, attributes
        FROM `{config.BQ_PROJECT_ID}.{config.BQ_DATASET_ID}.{config.BQ_TABLE_ID}`
        WHERE product_id IN ({placeholders})
    """
    rows = [dict(r) for r in client.query(query).result()]
    return {r["product_id"]: r for r in rows}


# ── Result formatting ─────────────────────────────────────────────────────────

def _format_results(
    hits: list[dict[str, Any]],
    details: dict[str, dict],
) -> str:
    if not hits:
        return "No results found."

    lines = ["| # | Score | Product ID | Title | Brand | Modality |",
             "|---|---|---|---|---|---|"]
    seen: set[str] = set()
    rank = 1
    for hit in hits:
        pid = hit.get("product_id", "?")
        if pid in seen:
            continue
        seen.add(pid)
        detail = details.get(pid, {})
        title    = detail.get("title", "—")
        brand    = detail.get("brand", "—")
        modality = hit.get("modality", "—")
        score    = f"{hit['score']:.4f}"
        lines.append(f"| {rank} | {score} | {pid} | {title} | {brand} | {modality} |")
        rank += 1

    return "\n".join(lines)


def _dedupe_and_fetch(hits: list[dict[str, Any]], top_k: int) -> str:
    # Deduplicate by product_id, keep highest score
    best: dict[str, dict] = {}
    for hit in hits:
        pid = hit["product_id"]
        if pid not in best or hit["score"] > best[pid]["score"]:
            best[pid] = hit
    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    details = _fetch_product_details([h["product_id"] for h in ranked])
    return _format_results(ranked, details)


# ── Public ADK tool functions ─────────────────────────────────────────────────

def search_by_text(query: str, top_k: int = 10) -> str:
    """Search the product index with a plain-text query.

    Searches both the text and image FAISS indexes, merges results,
    deduplicates by product_id, and returns a ranked markdown table.
    """
    logger.info("Text search: '%s' (top_k=%d)", query, top_k)
    embedding = embed_text(query)
    text_hits  = faiss_store.search_text(embedding, top_k=top_k)
    image_hits = faiss_store.search_image(embedding, top_k=top_k)
    all_hits   = text_hits + image_hits
    if not all_hits:
        return "Index is empty — run the indexing agent first."
    return _dedupe_and_fetch(all_hits, top_k)


def search_by_image(source: str, top_k: int = 10) -> str:
    """Search using an image file.

    Args:
        source: GCS URI (gs://…), HTTP URL, or local file path to an image.
        top_k:  Maximum number of results to return.
    """
    logger.info("Image search: '%s' (top_k=%d)", source, top_k)
    embedding = embed_image(source)
    hits = faiss_store.search_image(embedding, top_k=top_k)
    if not hits:
        return "No results found (or index is empty)."
    return _dedupe_and_fetch(hits, top_k)


def search_by_audio(source: str, top_k: int = 10) -> str:
    """Search using an audio file (Whisper → CLIP text).

    Args:
        source: GCS URI (gs://…), HTTP URL, or local file path to an audio file.
        top_k:  Maximum number of results to return.
    """
    logger.info("Audio search: '%s' (top_k=%d)", source, top_k)
    embedding, transcript = embed_audio(source)
    if not embedding.any():
        return "Could not transcribe the audio file."
    hits = faiss_store.search_text(embedding, top_k=top_k)
    if not hits:
        return f"No results found for transcript: '{transcript[:100]}'"
    header = f"**Transcript:** {transcript[:200]}\n\n"
    return header + _dedupe_and_fetch(hits, top_k)


def search_by_video(source: str, top_k: int = 10) -> str:
    """Search using a video file (sampled frames → mean CLIP image embed).

    Args:
        source: GCS URI (gs://…), HTTP URL, or local file path to a video file.
        top_k:  Maximum number of results to return.
    """
    logger.info("Video search: '%s' (top_k=%d)", source, top_k)
    embedding = embed_video(source)
    if not embedding.any():
        return "Could not extract frames from the video file."
    hits = faiss_store.search_image(embedding, top_k=top_k)
    if not hits:
        return "No results found."
    return _dedupe_and_fetch(hits, top_k)
