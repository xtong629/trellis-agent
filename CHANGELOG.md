# Changelog

All notable changes to Trellis. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Experiment numbers are from the committed
reports in `evals/reports/`; see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for method and limits.

## [0.2.0] - 2026-11-20

Second version: reflection, judge calibration, and three pre-registered experiments. Still a research
prototype. Experiment method, decision rules and limits: [`docs/EXPERIMENTS_V02.md`](docs/EXPERIMENTS_V02.md).

### Added
- **Reflection** (`trellis_agent.reflect`, PR #17): `ReflectingAgent` wraps an `Agent`; after a final
  answer a `Reflector` checks it and, if rejected, the agent retries with the reason (default: one
  retry, not re-checked). `LLMReflector` sees only the task, the tool steps and the answer. Every check
  and retry is counted in usage, cost and trace; a failing checker never fails the run. Opt-in; the
  default `Agent` is unchanged. Example: `examples/06_reflection.py`. A check that makes no model call
  no longer turns the run's cost into "unknown" (PR #27).
- **Judge calibration statistics** (`trellis_agent.eval.calibrate`, PR #19): 2x2 counts, agreement rate,
  Cohen's kappa and a seeded bootstrap interval.
- **v0.2 experiment tooling** (PRs #15-#25): task sets C1 (`exp-c1-reflect`) and C2 (`exp-c2-latency`) and
  the C4 answer set, frozen under `FROZEN.json` key `v0.2`; a trial/full-run driver with token and cost
  caps; slow, fault-injecting fake tools with a deterministic fault schedule; a results script that
  applies the pre-registered decision rules; trial reports kept in `evals/reports/trials/`.
- **Terminal demo** (PR #20): `scripts/demo.py` records offline commands to `docs/demo/` (transcript,
  asciicast, GIF).

### Experiments (one model, `glm-5.3-flash`, `reasoning_effort=low`; costs are list-price reference values)
- **C1, reflection: not run as designed.** The baseline passed every benefit task (1.000±0.000 over 5
  repeats), above the pre-registered band, so whether reflection helps is not decided. Reflection arm
  0.978±0.030; cost 0.3133 vs 0.2538 CNY (about 1.23x); 2 fixes and 2 harms. Both harms trace to an
  ambiguous wording in the frozen task set, which is disclosed and not changed (`exp_c1_*.json`).
- **C2, tool latency: hypothesis supported.** With 10 s simulated tool latency the planner's median time
  per task was 0.71 times the single agent's (34.50 vs 48.87 s), lower in 3 of 3 repeats; it was not
  faster at 1 s (ratio 1.47) or 4 s (0.95). Only 3 repeats per cell. The planner cost 2.2 to 3.1 times as
  much in every cell. With 30% transient tool failures (4 s) neither system was clearly better
  (`exp_c2_*.json`).
- **C4, judge calibration: hypothesis supported.** On 840 committed answers the judge matched the rule
  scorers on 839 (kappa 0.992, 95% CI 0.974-1.000). No human labels; the judge is the same model as the
  one under test; the answers come from simple tasks that rules can decide (`exp_c4_judge.json`).

### Unchanged
- The v2 task sets of experiments A and B and every v0.1.0 report.

### Not implemented
- Unchanged from 0.1.0 (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)); forgetting is still not
  called by the agent loop.

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
