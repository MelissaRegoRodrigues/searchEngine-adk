from tkinter import constants

from google.adk.agents.llm_agent import Agent

from . import prompt
from .config import config
from .sub_agents.comparison.agent import comparison_root_agent
from .sub_agents.keyword_finding.agent import keyword_finding_agent
from .sub_agents.search_results.agent import search_results_agent

root_agent = Agent(
    model=config.MODEL,
    name=config.AGENT_NAME,
    description=config.DESCRIPTION,
    instruction=prompt.ROOT_PROMPT,
    sub_agents=[
        keyword_finding_agent,
        search_results_agent,
        comparison_root_agent,
    ],
)