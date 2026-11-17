"""Example 01: the smallest possible agent.

python examples/01_single_agent.py --mock      # offline, deterministic
python examples/01_single_agent.py             # real model, reads LLM_* from env or .env
"""

import asyncio

from _common import parse_args, pick

from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.llm import ScriptedLLM
from trellis_agent.types import Message


async def main() -> None:
    args = parse_args(__doc__ or "")
    llm = pick(args.mock, ScriptedLLM([Message.assistant("I am Trellis demo assistant.")]))
    agent = Agent(llm, config=AgentConfig(system_prompt="You are a concise assistant."))
    result = await agent.arun("Hello, who are you?")
    print(f"Answer: {result.content}")
    print(f"tokens: {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out")
    print(f"stop_reason: {result.stop_reason}")


if __name__ == "__main__":
    asyncio.run(main())
