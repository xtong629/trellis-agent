"""Example 03: cost accounting and a hard budget.

Mock mode uses made-up DEMO prices so the arithmetic is easy to follow; they are NOT real prices.
For a real model, set PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M (per million tokens, from your
provider's pricing page), PRICE_CURRENCY (e.g. CNY) and BUDGET (an amount in that currency).
A model without a price is rejected ("fail closed"): a budget cannot be enforced on unknown prices.

    python examples/03_cost_and_budget.py --mock
"""

import asyncio
import os

from _common import check_llm_env, parse_args, real_client

from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import LLMClient, RuleLLM
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.types import Message, ToolCall, Usage


@tool
def lookup(topic: str) -> str:
    """Look something up."""
    return f"notes about {topic}"


async def main() -> None:
    args = parse_args(__doc__ or "")
    llm: LLMClient
    if args.mock:
        print("Using DEMO prices (not real): 1.00 in / 2.00 out DEMO per million tokens")
        meter = CostMeter(
            PriceTable({"mock": ModelPrice(1.0, 2.0, currency="DEMO")}), budget=1.5, currency="DEMO"
        )
        # Every call "uses" 400k in + 100k out = 0.60 DEMO; the model never stops using tools.
        llm = RuleLLM(
            lambda m, t: Message.assistant(
                None, (ToolCall(f"c{len(m)}", "lookup", {"topic": "x"}),)
            ),
            usage_per_call=Usage(400_000, 100_000),
            meter=meter,
        )
    else:
        check_llm_env()
        model = os.environ["LLM_MODEL"]
        try:
            currency = os.environ["PRICE_CURRENCY"]
            price = ModelPrice(
                float(os.environ["PRICE_INPUT_PER_M"]),
                float(os.environ["PRICE_OUTPUT_PER_M"]),
                currency=currency,
            )
        except KeyError as exc:
            raise SystemExit(f"set {exc.args[0]} to enable cost tracking") from exc
        meter = CostMeter(
            PriceTable({model: price}), budget=float(os.environ["BUDGET"]), currency=currency
        )
        llm = real_client(meter)

    agent = Agent(llm, tools=ToolRegistry([lookup]), config=AgentConfig(max_steps=20))
    result = await agent.arun("Research everything about the topic.")
    print(f"stop_reason: {result.stop_reason}")
    print(f"model calls: {meter.calls}")
    print(f"tokens: {meter.usage.prompt_tokens} in / {meter.usage.completion_tokens} out")
    if result.error:
        print(f"spent: {meter.total:.6f} of {meter.budget:.6f} {meter.currency}")
    else:
        print(f"spent: {meter.total:.6f} {meter.currency}")


if __name__ == "__main__":
    asyncio.run(main())
