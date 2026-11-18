# Changelog

All notable changes to Trellis. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Experiment numbers are from the committed
reports in `evals/reports/`; see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for method and limits.

## [0.1.0] - 2026-11-18

First version: a research prototype, not a production library.

### Added

**M0 - scaffold**
- Project layout, lint (ruff), type check (mypy), tests with the network disabled, CI on Python 3.11
  and 3.12 with actions pinned to commit SHAs.

**M1 - agent core**
- A single tool-calling loop (`Agent.stream`; `arun` and `run` collect it), per-session history,
  `max_steps`, failures reported in `RunResult` instead of raised.
- Tools with JSON Schema from type hints, argument validation, and errors returned to the model.
- Structured trace events and aggregate summaries.
- OpenAI-compatible client (missing `usage`, malformed tool arguments and HTTP failures handled) and
  scripted / rule-based mock models.
- Cost accounting from a user-provided price table.

**M2 - evaluation and cost control**
- Versioned, content-hashed task sets; 11 rule scorers plus an LLM judge; a runner with repeats and
  concurrency; reports; a baseline regression gate (`compare`, exit codes 0 / 1 / 2); a CLI.
- Budgets with a hard stop that also counts the call that crossed the line; agent and judge cost
  reported separately.

**M3 - planning and scheduling**
- LLM planner whose output is validated (shape, dependencies, cycles, size, worker kinds) and sent
  back for repair; event-driven scheduler with dependencies, parallelism, retry with backoff, failure
  policies (skip dependents, fail fast, replan) and budget stops; role-specific workers; `PlannedAgent`.

**M4 - memory and retrieval**
- Working window, episodic and semantic memory, relevance + recency + importance scoring, newer facts
  superseding older ones, write admission, consolidation and forgetting (methods; not called
  automatically).
- SQLite FTS5 BM25 retrieval with Chinese segmentation (jieba), an optional vector index, and RRF
  fusion. Only a deterministic hash embedder is included.

**Provider and cost details (PRs #1-#4)**
- Generic extra request parameters (for example `reasoning_effort`) on the client, agent config and
  CLI; `reasoning_tokens` recorded and shown as a share of output.
- Prices carry a currency and provenance (source, check date); budgets are an amount plus a currency;
  per-currency totals are kept apart; conversion only with an FX rate declared in the price file.
- `--meta` labels on the eval CLI, recorded in the report config.

**M5 - experiments (PRs #5-#11)**
- Task sets A (planner, 30 tasks) and B (memory, 20 scripts), fictional data, deterministic scoring,
  frozen by content hash; v1 to v2 changes after a 19-task pilot are documented, hashes of both kept.
- Resumable full-run driver, full-history reference condition, a generator for the result tables and
  a supplementary fanout timing run.
- Results on one model (`glm-5.3-flash`, `reasoning_effort=low`), 5 repeats:
  - A: single agent 0.987±0.018 and planner 0.993±0.015 success; planner cost 0.4877 vs 0.2331 CNY at
    list price (`exp_a_single.json`, `exp_a_planned.json`).
  - B: sliding window 0.400±0.035, tiered memory 0.980±0.027, full history (reference) 0.980±0.027
    (`exp_b_window.json`, `exp_b_memory.json`, `exp_b_full.json`).
  - Fanout-only timing (24 runs per system, concurrency 1): mean 13.23 s per task for the single agent
    and 27.36 s for the planner (`exp_a_fanout_timing.json`); a supplementary measurement.
  - One model, fictional tasks, small samples, no significance tests; costs are list-price reference
    values, not actual spend.

### Not implemented
- Hooks, guardrails, agent handoff, token streaming, persistent sessions, MCP, a real embedding
  client, tiered pricing, per-task budgets, `reasoning_content` pass-back. See
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
