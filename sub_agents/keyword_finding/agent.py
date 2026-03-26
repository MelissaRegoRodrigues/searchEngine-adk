from google.adk.agents.llm_agent import Agent
from ...config import config
from ...tools import bq_connector
from . import prompt

keyword_finding_agent = Agent(
    model=config.MODEL,
    name="keyword_finding_agent",
    description="A helpful agent to find keywords",
    instruction=prompt.KEYWORD_FINDING_AGENT_PROMPT,
    tools=[
        bq_connector.get_product_details_for_brand,
    ],
)