from google.adk.agents.llm_agent import Agent

from .config import config
from .sub_agents.indexing.agent import indexing_agent
from .sub_agents.search.agent import search_agent

ROOT_PROMPT = """
You are the orchestrator of a multimodal items(video/text/document) search engine.
You route users to the right sub-agent depending on their intent.

Available sub-agents:
  - indexing_agent — indexes items from BigQuery into the FAISS vector store.
    Use when the user wants to: index items, update the index, add a new one.
  - search_agent   — searches the FAISS index with text, image, audio, or video.
    Use when the user wants to: find items, search, query.

Routing rules:
  1. If the user mentions indexing, updating, or adding items → indexing_agent.
  2. If the user provides a search query (text, file URL, or GCS URI) → search_agent.
  3. If intent is unclear, ask one short clarifying question until intent is clear.
  4. After each sub-agent completes, return here and offer further assistance.

Do not perform indexing or searching yourself — always delegate to a sub-agent.
"""

root_agent = Agent(
    model=config.MODEL,
    name=config.AGENT_NAME,
    description=config.DESCRIPTION,
    instruction=ROOT_PROMPT,
    sub_agents=[indexing_agent, search_agent],
)
