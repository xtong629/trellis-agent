"""Example 05: tiered memory. A fact told early is recalled after it has left the working window,
and a corrected value replaces the old one.

    python examples/05_memory.py --mock     # offline: rule-based fake model + rule-based extractor
    python examples/05_memory.py            # real model; an LLM extracts the facts (costs tokens)
"""

import asyncio

from _common import parse_args, real_client

from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.llm import LLMClient, RuleLLM
from trellis_agent.memory.extract import FactExtractor, LLMExtractor, RuleExtractor
from trellis_agent.memory.manager import MemoryConfig, MemoryManager
from trellis_agent.memory.store import MemoryStore
from trellis_agent.types import Message, ToolSchema


def mock_model(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """Answers from the memory note if there is one, otherwise says it does not know."""
    for m in messages:
        if m.role == "system" and (m.content or "").startswith("Relevant memories"):
            fact = (m.content or "").splitlines()[1].removeprefix("- ")
            return Message.assistant(f"我记得:{fact}")
    return Message.assistant("好的")


async def main() -> None:
    args = parse_args(__doc__ or "")
    mock = RuleLLM(mock_model) if args.mock else None
    llm: LLMClient = mock or real_client()
    extractor: FactExtractor = RuleExtractor() if args.mock else LLMExtractor(llm)
    memory = MemoryManager(MemoryStore(), extractor, MemoryConfig(working_token_budget=60))
    agent = Agent(
        llm, memory=memory, config=AgentConfig(system_prompt="You are a helpful assistant.")
    )

    turns = [
        "我的截止日期是周五",
        "闲聊第一句," + "天气不错," * 6,
        "闲聊第二句," + "今天很忙," * 6,
        "截止日期改到周二了",
        "闲聊第三句," + "周末去爬山," * 6,
    ]
    for text in turns:
        await agent.arun(text, session_id="demo")
    result = await agent.arun("我的截止日期是哪天?", session_id="demo")

    if mock is not None:
        sent = [m for m in mock.calls[-1].messages if m.role != "system"]
        stored = len(agent._sessions["demo"]) - 2  # history before the final question and answer
        print(f"older messages trimmed from the window: {stored - (len(sent) - 1)} of {stored}")
    note = next(
        (m.content or "" for m in result.messages if (m.content or "").startswith("Relevant")), ""
    )
    print(f"memory note: {note.splitlines()[1] if note else '(none)'}")
    print(f"answer: {result.content}")
    old = [i for i in memory.store.items(include_superseded=True) if i.superseded_by]
    print(f"superseded (kept as history, never recalled): {len(old)}")


if __name__ == "__main__":
    asyncio.run(main())
