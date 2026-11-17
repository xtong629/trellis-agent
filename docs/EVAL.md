# Evaluation and cost control

Trellis treats evaluation as the first-class feature: every other capability (planning, memory) is
meant to be *measured* with it. This page explains how it works and, just as important, what its
numbers do and do not mean.

```text
task set (YAML/JSON/JSONL, versioned, content-hashed)
        |
        v
  EvalRunner --- fresh target per (task, repeat) ---> Agent / any runnable
        |                                                    |
        |<------------ RunResult (answer, trace, usage, cost)
        v
  scorers (rules + optional LLM judge) -> CaseResult -> EvalReport (JSON, deterministic)
                                                           |
                       saved baseline <---- compare <------+   exit code 0 / 1 / 2
```

## Task sets

```yaml
name: smoke
version: 1            # bump it when you change tasks on purpose
tasks:
  - id: add-numbers
    input: "What is 2 + 3?"
    tags: [tools, math]
    scorers:
      - {type: number_match, value: 5}
      - {type: tool_called, name: add}
```

* A task set is a file (`.yaml`, `.yml`, `.json`, or `.jsonl` with a header line) or a directory
  containing `taskset.<ext>`.
* `content_hash` (SHA-256 of the *parsed* content) is stored in every report. Reformatting a file
  does not change it; editing, adding, removing or reordering a task does. A comparison between two
  different hashes is refused (exit 2): you would be comparing different questions.
* Typos are errors, not silent no-ops: unknown fields, unknown scorer types and unknown scorer
  parameters are rejected when the file is loaded, before any money is spent.
* `expect_success: false` marks tasks where the run is *supposed* to end in an error.

## Scorers

A task passes only if the built-in `run_succeeded` check **and** every listed scorer pass.

| Type | Parameters | Passes when |
| --- | --- | --- |
| `exact_match` | `value`, `strip`=true, `casefold`=false | output equals `value` |
| `contains` / `not_contains` | `value`, `casefold`=false | output does / does not contain `value` |
| `regex` | `pattern`, `ignore_case`=false | `re.search` finds a match |
| `number_match` | `value`, `tolerance`=0, `which`=any\|last | a number in the output equals `value` (handles `1,234.5`) |
| `tool_called` | `name`, `min_times`=1 | the trace has that many calls of the tool |
| `tool_forbidden` | `name` | the tool was never called |
| `max_tokens` | `value` | total agent tokens <= `value` |
| `max_cost_usd` | `value` | agent cost <= `value` (fails if the cost is unknown) |
| `max_latency_ms` | `value` | measured run duration <= `value` (fails if not measured) |
| `stop_reason` | `value`=end_turn | the run stopped for that reason |
| `llm_judge` | `rubric`, `pass_threshold`=0.7 | a judge model scores the answer >= threshold |

Scorers that *cannot* verify their condition (no price, no timing data) **fail with an
explanation**; they never pass silently. In Python you can add your own with `CustomScorer`.

### The LLM judge

`llm_judge` grades open-ended answers against a rubric. Its design choices, and why:

* temperature 0 and a fixed JSON reply format (`{"score": 0..1, "reason": "..."}`) for stability;
* strict parsing: an unparsable or out-of-range judgement is a *scorer error* (the case fails),
  never a pass; fenced or chatty JSON is tolerated;
* the judge's tokens and cost are **reported separately** from the agent's (below);
* it is skipped when the run itself failed (nothing to grade, nothing to pay for).

A judge is a second model with its own noise and bias. Prefer rule scorers whenever a rule can
decide, and treat judge scores as evidence, not truth.

## Cost accounting

Every figure in a report comes from the per-call `LLMResponse.cost_usd`, which the LLM client
computes from **your** price table (`--prices FILE`, USD per million tokens, optional cached-input
price). Trellis ships **no** real vendor prices; `evals/prices.demo.yaml` is **DEMO PRICES ONLY
(仅为示例,非真实价格)**: its numbers do not correspond to any real model or vendor and exist only
to price the offline mock models.

| Field | Meaning |
| --- | --- |
| `agent_cost_usd`, `usage` | what the system under test spent |
| `judge_cost_usd`, `judge_usage` | what LLM judges spent (separate, so judging cost is visible) |
| `cost_usd` | agent + judge |
| `cost_per_success` | `cost_usd / passed`: a better way to compare systems than raw cost |
| `cost_complete` | false if any call had no price; totals are then lower bounds |
| `usage.reasoning_tokens`, `reasoning_share` | hidden reasoning tokens (part of the output) and their share of it; present only when the provider reports them; a lower bound if only some calls did |
| `estimated_usage` | true if the provider reported no usage and tokens were estimated (chars/4) |
| `meter_total_usd`, `meter_discrepancy_usd` | the shared `CostMeter`'s own total, and report minus meter (should be 0) |

An unknown cost is `null`, never `0`.

### Budgets, and the call that crosses the line

A hard budget (`--budget-usd`, or `CostMeter(budget_usd=...)`) works at two levels:

1. **Per call.** `CostMeter.record` books the spend of every call, then raises `BudgetExceeded` if
   the total passed the budget. The call that crossed the line *has already been paid for*, so the
   exception carries it: `BudgetExceeded.response` is the full `LLMResponse` (message, usage, cost).
2. **Per job.** The runner will not start a new (task, repeat) job once the spend so far reaches the
   budget.

What happens to the over-budget call depends on who made it:

* **Agent call:** the agent stops (`stop_reason="budget_exceeded"`), does *not* use the reply, but
  *does* add its tokens and cost to the run's `usage`/`cost_usd` and records a `model_request`
  trace event flagged `over_budget`. The case is `aborted`.
* **Judge call:** the judge's cost is attached to the case as a judge cost, the case is `aborted`.

Consequently the report's cost equals the meter's cost, including the final over-budget call
(`meter_discrepancy_usd` is 0). The runner's job-level check is soft: with concurrency N, up to N
jobs already in flight may finish after the budget is reached, so the final spend can exceed the
budget by about N cases. Use `--concurrency 1` for the tightest cap.

Case statuses: `passed`, `failed`, `aborted` (budget hit mid-case), `not_run` (budget hit earlier).
Pass rates are computed over `passed + failed` only; the report states how many jobs were aborted
or not run, and `compare` / `save-baseline` refuse incomplete reports.

A budget **requires prices** for every model in use (the CLI checks up front): a limit cannot be
enforced on calls whose price is unknown.

## Determinism

With a deterministic target (the mock models), two runs of the same task set produce
**byte-identical** report files, even at different concurrency (apart from the recorded
`concurrency` setting itself):

* the saved JSON has sorted keys, fixed formatting, real unicode, no timestamps;
* cases are ordered by (repeat, task) regardless of which finished first;
* sums use `math.fsum`, so they do not depend on the order concurrent calls finished in;
* wall-clock timing is left out unless you pass `--include-timing`.

This is checked in the test suite and by the CI smoke step. With a *real* model, outputs vary run
to run; use `--repeats` to measure that spread (`pass_rate_by_repeat`, mean and standard deviation).

## Regression gate

```bash
python -m trellis_agent.eval run evals/smoke --llm mock --prices evals/prices.demo.yaml --out runs/smoke.json
python -m trellis_agent.eval save-baseline runs/smoke.json            # -> evals/baselines/smoke.json
python -m trellis_agent.eval compare evals/baselines/smoke.json runs/smoke.json
```

| Exit code | Meaning |
| --- | --- |
| 0 | no regression within the configured limits |
| 1 | a gate failed: a task regressed, the pass rate dropped too far, or cost rose too much |
| 2 | cannot compare (task set changed, incomplete run, cost limit set but cost unknown) or bad input |
| 3 | (`run` only) aborted by its budget; results are partial |

Gates: any regressed task fails unless `--max-regressed-tasks N` allows it; `--max-pass-drop F`
(fraction) bounds the overall drop; `--max-cost-increase-pct P` is optional. When a gate fails, the
output names the task, the failing check, the reason, and what the system actually said:

Real output (abridged) from running the smoke set against a deliberately broken target that
answers everything with "The answer is 6":

```text
COMPARE FAILED: 8 tasks regressed; pass rate dropped 100.0 pp (limit 0.0 pp)

baseline: smoke v1 [17adf4bed184]
current:  smoke v1 [17adf4bed184]

pass rate : 100.0% -> 0.0% (-100.0 pp; limit 0.0 pp)
cost      : $0.002510 -> $0.000350 (-86.1%; limit none)
tasks     : 8 compared, 8 regressed, 0 improved

REGRESSIONS
  - add-numbers: passed 1/1 -> 0/1
      (repeat 0) failed check number_match(5): no number equal to 5 (+/-0); numbers in output: ['6']
      (repeat 0) failed check contains('answer is 5'): output does not contain 'answer is 5'
      (repeat 0) failed check tool_called('add'): tool 'add' called 0 time(s), expected at least 1; tools called: none
      (repeat 0) output: 'The answer is 6'
  ...
exit code 1
```

## What these numbers do not mean

* **Mock results prove the machinery, not model quality.** The demo model is a rule-based fake.
* A small task set measures *those tasks*. Report the sample size next to any rate.
* Costs are only as accurate as the price table you supply, and are estimates, not billing records.
* Judge scores are noisy; one repeat of one task says little about a real model.
