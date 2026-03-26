from asyncio import constants
from scipy import constants

from google.adk.agents.llm_agent import Agent

from ...config import config
from . import prompt

comparison_generator_agent = Agent(
    model=config.MODEL,
    name="comparison_generator_agent",
    description="A helpful agent to generate comparison.",
    instruction=prompt.COMPARISON_AGENT_PROMPT,
)

comparsion_critic_agent = Agent(
    model=config.MODEL,
    name="comparison_critic_agent",
    description="A helpful agent to critique comparison.",
    instruction=prompt.COMPARISON_CRITIC_AGENT_PROMPT,
)

comparison_root_agent = Agent(
    model=config.MODEL,
    name="comparison_root_agent",
    description="A helpful agent to compare titles",
    instruction=prompt.COMPARISON_ROOT_AGENT_PROMPT,
    sub_agents=[comparison_generator_agent, comparsion_critic_agent],
)