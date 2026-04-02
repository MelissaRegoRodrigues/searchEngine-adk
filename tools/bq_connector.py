# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigQuery connector tool used by the keyword_finding_agent.

The tool queries a BigQuery table that stores product data and returns
rows matching the requested brand as a markdown table so the LLM can
read them directly.

Expected BigQuery table schema
──────────────────────────────
Column        Type     Description
brand         STRING   Brand name (case-insensitive match is applied)
title         STRING   Product listing title
description   STRING   Long-form product description
attributes    STRING   JSON string, e.g. {"Size":"10","Color":"Blue"}

Environment variables (set via .env or your shell):
  BQ_PROJECT_ID  – GCP project id
  BQ_DATASET_ID  – BigQuery dataset id
  BQ_TABLE_ID    – BigQuery table id
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from google.cloud import bigquery  # type: ignore
    _BQ_AVAILABLE = True
except ImportError:
    _BQ_AVAILABLE = False
    logger.warning(
        "google-cloud-bigquery is not installed – BigQuery calls will use "
        "mock data.  Run: pip install google-cloud-bigquery"
    )

from ..config import config

# ---------------------------------------------------------------------------
# Mock data – returned when BigQuery is unavailable or BQ_PROJECT_ID has not
# been configured. Replace / extend this list with real sample products.
# ---------------------------------------------------------------------------
_MOCK_PRODUCTS: list[dict[str, str]] = [
    {
        "brand": "MockBrand",
        "title": "Kids' Joggers",
        "description": (
            "Comfortable and supportive running shoes for active kids. "
            "Breathable mesh upper keeps feet cool, while the durable "
            "outsole provides excellent traction."
        ),
        "attributes": json.dumps({"Size": "10 Toddler", "Color": "Blue/Green"}),
    },
    {
        "brand": "MockBrand",
        "title": "Adult Running Shoes",
        "description": (
            "High-performance running shoes designed for road and trail. "
            "Lightweight foam midsole for responsive cushioning."
        ),
        "attributes": json.dumps({"Size": "10", "Color": "Black/White"}),
    },
    {
        "brand": "MockBrand",
        "title": "Casual Sneakers",
        "description": (
            "Everyday sneakers with a clean, minimalist design. "
            "Padded collar and cushioned insole for all-day comfort."
        ),
        "attributes": json.dumps({"Size": "9", "Color": "White"}),
    },
]


def _rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    """Convert a list of row dicts to a markdown table string."""
    if not rows:
        return "No products found."
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def get_product_details_for_brand(brand: str) -> str:
    """Fetches product details (title, description, attributes) for a brand.

    Args:
        brand: The brand name to look up (case-insensitive).

    Returns:
        A markdown-formatted table of products, or an error message.
    """
    # Use mock data when BigQuery is not configured
    if not _BQ_AVAILABLE or config.BQ_PROJECT_ID == "your-gcp-project":
        logger.info("Using mock product data for brand: %s", brand)
        results = [
            p for p in _MOCK_PRODUCTS
            if p["brand"].lower() == brand.lower()
        ]
        # Fall back to all mock rows so the agent always has data to work with
        if not results:
            results = _MOCK_PRODUCTS
        return _rows_to_markdown(results)

    # Live BigQuery path
    try:
        client = bigquery.Client(project=config.BQ_PROJECT_ID)
        table_ref = (
            f"`{config.BQ_PROJECT_ID}"
            f".{config.BQ_DATASET_ID}"
            f".{config.BQ_TABLE_ID}`"
        )
        query = f"""
            SELECT brand, title, description, attributes
            FROM {table_ref}
            WHERE LOWER(brand) = LOWER(@brand)
            LIMIT 50
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("brand", "STRING", brand)
            ]
        )
        rows = [dict(r) for r in client.query(query, job_config=job_config).result()]
        logger.info("BigQuery returned %d rows for brand '%s'", len(rows), brand)
        return _rows_to_markdown(rows)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("BigQuery query failed: %s", exc)
        return f"Error querying BigQuery: {exc}"
