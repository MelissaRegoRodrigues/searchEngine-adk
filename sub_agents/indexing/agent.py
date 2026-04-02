from google.adk.agents.llm_agent import Agent

from ...config import config
from ...tools.indexing_tools import index_new_products, get_index_stats

INDEXING_AGENT_PROMPT = """
You are the indexing agent for a multimodal items(search engine.
Your job is to keep the FAISS vector index up to date with the latest products
from BigQuery.

Steps:
1. Ask the user which item they want to index (if not already provided).
2. Call `get_index_stats` to show the current index status before indexing.
3. Call `index_new_products` with the item name.
   - This tool will skip items already in the index (incremental mode).
   - It generates CLIP embeddings for text, images, audios, and videos.
4. Show the returned summary to the user.
5. Call `get_index_stats` again to confirm the new totals.
6. Transfer back to root_agent.

Key constraints:
- Do not attempt to generate embeddings yourself.
- If the tool reports errors, relay them clearly to the user.
- Never fabricate indexing results.
"""

indexing_agent = Agent(
    model=config.MODEL,
    name="indexing_agent",
    description=(
        "Reads items from BigQuery and indexes them incrementally into the "
        "FAISS store using CLIP embeddings (text, image, audio, video)."
    ),
    instruction=INDEXING_AGENT_PROMPT,
    tools=[index_new_products, get_index_stats],
)
