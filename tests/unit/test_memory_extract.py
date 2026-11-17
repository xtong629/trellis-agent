import json

import pytest

from trellis_agent.cost import BudgetExceeded, CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import ScriptedLLM
from trellis_agent.memory.extract import Fact, LLMExtractor, RuleExtractor
from trellis_agent.types import Message, Usage


async def test_rule_extractor_pulls_keyed_facts_from_user_statements() -> None:
    turn = [Message.user("我的截止日期是周五。今天天气不错。"), Message.assistant("好的")]
    extraction = await RuleExtractor().extract(turn)
    assert extraction.facts == [Fact("我的截止日期是周五", key="截止日期", importance=0.7)]
    assert extraction.usage == Usage() and extraction.cost_usd == 0.0  # rules cost nothing


async def test_an_update_gets_the_same_key_so_it_can_supersede_the_old_value() -> None:
    old = await RuleExtractor().extract([Message.user("我的截止日期是周五")])
    new = await RuleExtractor().extract([Message.user("截止日期改到周二了")])
    assert old.facts[0].key == new.facts[0].key == "截止日期"
    eng = await RuleExtractor().extract([Message.user("My deadline is Friday.")])
    assert eng.facts[0].key == "deadline"


@pytest.mark.parametrize(
    "text", ["我的截止日期是哪天?", "What is my deadline?", "截止日期是什么", "ok"]
)
async def test_questions_and_chatter_are_not_facts(text: str) -> None:
    assert (await RuleExtractor().extract([Message.user(text)])).facts == []


async def test_only_user_messages_are_mined_by_rules() -> None:
    turn = [Message.assistant("我的截止日期是周五")]
    assert (await RuleExtractor().extract(turn)).facts == []


def llm_with(reply: str, meter: CostMeter | None = None) -> ScriptedLLM:
    return ScriptedLLM(
        [Message.assistant(reply)], usage_per_call=Usage(1_000_000, 0), meter=meter, model="m"
    )


async def test_llm_extractor_parses_facts_and_accounts_for_its_own_cost() -> None:
    reply = json.dumps(
        {
            "facts": [
                {"text": "User is allergic to peanuts", "key": "allergy", "importance": 0.95},
                {"text": "Prefers tea", "key": None, "importance": 7},  # clamped to 1.0
                {"text": "", "importance": 0.5},  # skipped: empty
                {"key": "x"},  # skipped: no text
            ]
        }
    )
    meter = CostMeter(PriceTable({"m": ModelPrice(1.0, 0.0)}))
    llm = llm_with(reply, meter)
    out = await LLMExtractor(llm).extract(
        [Message.user("I am allergic to peanuts"), Message.assistant("noted")]
    )
    assert out.facts == [
        Fact("User is allergic to peanuts", "allergy", 0.95),
        Fact("Prefers tea", None, 1.0),
    ]
    assert out.usage == Usage(1_000_000, 0) and out.cost_usd == pytest.approx(1.0)
    assert meter.by_tag == {"memory_extract": pytest.approx(1.0)}
    prompt = "\n".join(m.content or "" for m in llm.calls[0].messages)
    assert "allergic to peanuts" in prompt and "noted" in prompt and llm.calls[0].temperature == 0.0


async def test_an_unusable_reply_yields_no_facts_but_the_spend_is_still_reported() -> None:
    out = await LLMExtractor(llm_with("sorry, no idea")).extract([Message.user("hi there friend")])
    assert out.facts == [] and out.usage == Usage(1_000_000, 0)


async def test_budget_exceeded_propagates_with_the_paid_response() -> None:
    meter = CostMeter(PriceTable({"m": ModelPrice(1.0, 0.0)}), budget_usd=0.1)
    with pytest.raises(BudgetExceeded) as info:
        await LLMExtractor(llm_with('{"facts": []}', meter)).extract([Message.user("hello")])
    assert info.value.response is not None
