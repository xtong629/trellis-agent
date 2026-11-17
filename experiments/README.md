# Experiments

Two controlled comparisons, designed before any real model call. Not part of the installed library.

| | Question | Systems compared | Task set |
| --- | --- | --- | --- |
| A | Does planning (planner + workers + retry) beat one agent with all tools? | `experiments.targets:a_single` vs `a_planned` | `evals/exp_a_planner` (30 tasks) |
| B | Does tiered memory beat a sliding window of the same size? | `b_window` vs `b_memory` | `evals/exp_b_memory` (20 scripts) |

Design rules: all data is fictional (nothing can be answered from world knowledge); expected answers
are computed from that data; scoring uses deterministic rules only (no LLM judge); every set has
families that should benefit and families that should not, so a result can also show *no* effect.
The families and their hypotheses are in each task's `metadata`.

* A families: `serial` (price, then rate, then multiply), `fanout` (independent lookups, then a sum),
  `flaky` (the exchange-rate service is busy on its first call), `single` (one lookup; no benefit expected).
* B families: `allergy` and `deadline` (early fact, later asked; deadline scripts include a
  correction), `refund-zh` (Chinese), `fits` (fact still inside the window; no benefit expected),
  `irrelevant` (the last question needs nothing remembered; memory only costs tokens).
* B's baseline uses the same working window (`WINDOW_TOKENS`); memory's extra model calls (fact
  extraction) are counted in its cost.
* B scripts are multi-turn: turns are separated by a `---` line in `input`; only the last reply is scored.

The task set files are generated: `python -m experiments.build`. `FROZEN.json` records their content
hashes. After the freeze, only bug fixes are allowed; each one updates the hash and triggers a full re-run.

Offline check (mock models, proves the machinery only): each of the four targets runs against its
task set twice and produces byte-identical reports (`tests/integration/test_experiments_pipeline.py`).
