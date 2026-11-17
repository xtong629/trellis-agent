# Trellis

[![CI](https://github.com/xtong629/trellis-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/xtong629/trellis-agent/actions/workflows/ci.yml)

> A small agent framework where planning and memory are **measurable**: eval-first, cost-capped, mock-testable.

**Status: early development.** The agent core and the evaluation harness work and are tested.
Planning/scheduling and tiered memory are implemented; the experiments that show whether they
actually help come next. See [Roadmap](#roadmap) and [Limitations](#limitations).

## Why

Most agent demos show that something *can* work. Trellis is built around the harder question:
*did this change make the system better, and what did it cost?*

* **Eval-first.** Versioned, content-hashed task sets; rule and LLM-judge scorers; a baseline
  regression gate whose failures say which task failed, which check, and why.
* **Cost-capped.** Per-call cost accounting from *your* price table, agent and judge cost reported
  separately, a hard budget that also counts the call that crossed it.
* **Mock-testable.** Deterministic mock models make the whole stack testable offline; the test
  suite blocks the network entirely. Two runs of an evaluation are byte-identical.

## What works today

| Area | What you get |
| --- | --- |
| Agent loop | one event loop (`stream`), `arun`/`run` collect it; tool calling with per-session history |
| Tools | JSON Schema from type hints; arguments validated; errors fed back to the model, never crash the run |
| LLM clients | OpenAI-compatible client (handles missing `usage`, malformed tool arguments, HTTP failures); scripted/rule mocks |
| Cost | `PriceTable`, `CostMeter`, hard budgets, order-independent exact sums |
| Tracing | structured events, per-run traces, aggregate summaries |
| Evaluation | task sets, 11 scorer types + `llm_judge`, runner with repeats and concurrency, reports, baseline `compare`, CLI |
| Memory | working window within a token budget; episodic and semantic memories with relevance + recency decay + importance scoring; newer facts supersede older ones; write admission (duplicates, per-turn limit); consolidation and forgetting; optional, so an agent without memory behaves exactly as before |
| Retrieval | jieba-segmented Chinese + stemmed English full-text search on SQLite FTS5 (BM25, OR-ed terms); optional vector index fused with RRF |
| Planning | LLM planner whose plans are validated and sent back for repair; event-driven scheduler with dependencies, parallelism, retry with backoff, failure policies (skip dependents / fail fast / replan), budgets; role-specific worker agents dispatched by task kind |

Details of the evaluation harness, cost rules and the budget mechanism: [`docs/EVAL.md`](docs/EVAL.md).

## Quick start

Requires Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                        # offline; the network is blocked during tests
python examples/02_tools_agent.py --mock      # a tool-using agent, no API key needed
python examples/03_cost_and_budget.py --mock  # cost accounting and a hard budget
python examples/04_planned.py --mock          # planner + researcher/calculator worker agents
python examples/05_memory.py --mock           # recall after the working window, updated facts
```

Run the evaluation smoke suite and the regression gate:

```bash
python -m trellis_agent.eval run evals/smoke --llm mock \
    --prices evals/prices.demo.yaml --out runs/smoke.json
python -m trellis_agent.eval compare evals/baselines/smoke.json runs/smoke.json
```

Provider-specific request fields (for example a reasoning level) can be set per run with
`--extra-param reasoning_effort=low` (repeatable, JSON values allowed); they are sent as top-level
body fields and recorded in the report config.

To use a real model, copy `.env.example` to `.env`, set `LLM_BASE_URL`, `LLM_API_KEY` and
`LLM_MODEL` (any OpenAI-compatible endpoint), and drop `--mock` / use `--llm openai`. Real runs need
a price file for your model. `evals/prices.demo.yaml` shows the format; its numbers are
**DEMO PRICES ONLY (仅为示例,非真实价格)** and do not correspond to any real model or vendor.

## Results

**No experiment results yet.** The planned comparisons (planner on/off, tiered memory on/off, same
tasks, success rate / tokens / cost) have not been run. Numbers will be added here only after they
have been measured, with the task set hash, model, price table and sample size next to them.
Offline mock runs only exercise the code; they say nothing about real-model performance.

## Roadmap

| Milestone | Scope | State |
| --- | --- | --- |
| M0 | project scaffold, lint/type/test/CI | done |
| M1 | agent loop, tools, tracing, cost accounting, LLM clients | done |
| M2 | evaluation and cost control | done |
| M3 | LLM planner + event-driven scheduler with multi-role workers | done |
| M4 | tiered memory + Chinese/English hybrid retrieval | done |
| M5 | experiments and write-up | planned |

## Project layout

```text
src/trellis_agent/   agent, tools, llm/, cost, trace, eval/, planner/, memory/, retrieval/
tests/               unit/ and integration/ (fake OpenAI-compatible HTTP server for client contracts)
evals/               smoke task set, demo prices, committed baseline
examples/            runnable examples (all support --mock)
docs/                EVAL.md
```

## Limitations

Known, accepted, and tested where possible:

* **Scheduler.** A task that is waiting to retry still occupies a concurrency slot. Only a global
  budget exists (the shared `CostMeter`); there is no per-task budget. Single process, no
  persistence, no per-task timeout.
* **Stemming.** The Porter stemmer maps `verify` to `verifi` and `verification` to `verif`, so a
  search for "verify" does not find "verification" (an `xfail` test records it). Prefix matching
  would not fix `retries`/`retry` and would add noise.
* **Chinese segmentation.** jieba splits words it does not know: "单号" becomes "单" + "号". Domain
  terms need a user dictionary (`Tokenizer(user_dict=...)`).
* **Token estimates.** The working window and estimated embedding usage count characters / 4,
  which under-counts Chinese text, so the real token count of a window can exceed its budget.
* **Recall.** Only semantic memories are recalled; raw episodes become recallable once
  consolidated. A consolidated summary is not reconciled with later corrections, so it can still
  contain an old value.
* **Scope.** Facts are shared by everything that uses one `MemoryManager`; they are not scoped
  per user or per session (the session id is stored but not used to filter).
* **Extraction.** `RuleExtractor` is a toy ("X是Y" / "my X is Y" statements); `LLMExtractor` costs
  tokens (counted and shown in the trace) and can miss or invent facts.
* **Vectors.** The vector index is in-memory, pure Python and not persisted; only a deterministic
  hash embedder is included (no real embedding client yet), and embedding cost is an estimate.
* **Mocks.** All offline runs use rule-based fake models. They exercise the code and say nothing
  about real-model quality; no experiment results exist yet.

## Acknowledgements

Trellis is inspired by [LightAgent](https://github.com/wanxingai/LightAgent) (Apache-2.0). I studied
its design (reference commit `f8083ab`) and am writing an independent implementation focused on
evaluation, planning/scheduling and tiered memory. Trellis is not affiliated with or endorsed by the
LightAgent authors. See [`NOTICE`](NOTICE).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
