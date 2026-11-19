# Experiments in v0.2

This page holds the v0.2 experiments. Each one is **pre-registered here (hypothesis, design,
decision rules) and merged before any real model call**. Results are added afterwards in a separate
"Results" subsection and never change the pre-registered text above it. The v0.1.0 experiments and
their reports (`docs/EXPERIMENTS.md`, `evals/reports/exp_a_*`, `exp_b_*`) are not modified.

## Rules shared by every v0.2 experiment

* **Model and settings:** `glm-5.3-flash` through the OpenAI-compatible API, `reasoning_effort=low`,
  temperature 0, the same as v0.1.0. Prices from `evals/prices.glm.yaml` (CNY list prices); a free
  token package is used, so every cost is a **reference value at list price, not actual spend**.
* **New task sets only.** Each experiment has its own task set name and content hash, recorded in
  `experiments/FROZEN.json` under the key `v0.2`. The frozen v2 sets of experiments A and B are not
  changed, and no v0.1.0 number is reused for a different task set; baselines are run again.
* **Order:** pre-registration (this page, merged) -> baseline-only trial -> freeze (task set file and
  hash merged) -> implementation of the component under test -> full run -> results. Task wording is
  fixed before the component under test is written, so tasks cannot be shaped around it.
* **Scoring:** only the existing rule scorers, unchanged. Every task's scorer configuration is tested
  before the trial against several plausible reply shapes (answer line first or last, working with
  other numbers, bold, lower case, units, rounding to 2 decimals, thousands separators) that must pass,
  and the same shapes with wrong values that must fail (`tests/unit/test_experiments_c1.py`). Known
  limit, kept: a full-width colon (`ANSWER：`) is not read as the answer marker; tasks are in English.
* **Trials are not results.** Trial numbers are used only for the pre-registered band check and to
  look for scoring bugs; they are not reported as findings.
* **Run safety:** every run has a CNY cap and a token cap; the driver (`experiments/v02_run.py`) stops
  between chunks on a non-200 response, a call slower than 60 s, 80% of either cap, a case aborted by
  the budget, or a provider/target error. A wrong answer or `max_steps` is a result, not a stop.
  The whole of v0.2 stays under 5M tokens and 20 CNY at list price.
* **Honest reporting:** negative and inconclusive results are reported with the same prominence as
  positive ones. If a design condition is not met (for example the baseline is outside its band), the
  experiment is reported as **not run as designed**, with its numbers, and its hypothesis is not decided.

## C1 (R1): does one round of self-check and retry raise success?

**Question.** An agent answers; a checker (the same model) reviews the answer against the tool
results; if the checker rejects it, the agent tries once more with the checker's reason. Does this
raise the pass rate on multi-step tool tasks, at what cost, and does it break answers that were right?

**Systems** (both: one `Agent` with the six C1 tools, the same system prompt, `max_steps=12` per
attempt, the same model; concurrency 2 tasks at a time):

* `c1_baseline`: the agent alone.
* `c1_reflect`: the same agent wrapped in `ReflectingAgent` (to be added in `src/trellis_agent/reflect.py`).

**Reflection protocol** (fixed now; implemented after the freeze):

1. The agent runs normally. Only a run that ended with a final answer (`end_turn`) is checked;
   a run that ended in `max_steps`, a provider error or the budget is returned as it is.
2. The checker gets the task text, the final answer, and the run's tool calls with their results.
   It does **not** get the expected answer, the scorers, or anything task-set specific; its prompt is
   generic (check every step against the tool results, every condition stated in the task, the
   arithmetic, and the requested answer format). It replies with JSON `{"ok": bool, "reason": str}`.
   An unreadable verdict counts as `ok` (no retry) and is recorded.
3. If the verdict is not ok, the reason is sent to the agent as a new user turn in the same session,
   asking for a corrected final answer in the same format. The agent may use tools again.
4. At most one retry; the retry's answer is final and is not checked again.
5. All tokens and cost (check calls and the retry) are counted in the run's result; check calls are
   tagged `reflect`. Each case's first answer, verdict and final answer are written to a side file
   (`evals/reports/exp_c1_reflect_attempts.json`) so fixes and harms can be counted.

**Task set `exp-c1-reflect`** (`experiments/c1_tasks.py`), fictional catalog, 24 tasks, 6 per family:

| family | expected | what makes it hard |
| --- | --- | --- |
| deep | benefit | item -> supplier discount -> currency conversion that needs 2 or 3 hops (only neighbouring currencies have a direct rate) |
| constraint | benefit | cheapest item in a category among those meeting a stock condition; some items are priced per pack |
| trap | benefit | quantity in single units, price per pack |
| easy | no benefit | one lookup (control: measures harm) |

Every numeric answer ends with an `ANSWER: <number>` line and is scored with `number_match`
(`which: marked`, tolerance 0.01), plus `tool_called` checks for the tools a task needs.

**Difficulty knob (the only one).** `LEVEL` changes the `deep` and `constraint` families only:
level 0 (easier: one-hop conversion; no stock condition), level 1 (the design above), level 2
(harder: per-pack items in `deep`; in `constraint`, the cost of buying exactly k units in whole packs).

**Trial and band rule.**

* Trial: `c1_baseline` only, level 1, all 24 tasks, 1 repeat; caps 1 CNY and 0.3M tokens. Every
  reply is read for scoring bugs.
* Band: the baseline pass rate on the 18 benefit tasks must be in **[0.3, 0.7]**.
  * In band: freeze level 1.
  * Above 0.7: move to level 2; below 0.3: move to level 0. One move only, then a second
    baseline-only trial at the new level (same caps), and that level is frozen whatever it shows.
* A scoring bug found in the trial may be fixed in the task set (with a regression test) before the
  freeze and is documented here. A fix that would need a change to a library scorer (shared with the
  frozen v2 sets) is not made; the experiment stops and the problem is reported.

**Full run.** Both systems, 5 repeats, systems alternating per repeat, concurrency 2; caps 3 CNY and
2M tokens. The band is checked again on the full-run baseline (mean over 5 repeats, benefit tasks);
outside [0.3, 0.7], C1 is reported as **not run as designed** and H1 is not decided.

**Metrics.** Per system: pass rate (all, benefit, per family) as mean ± sample sd over 5 per-repeat
rates; cost (CNY, list price), tokens, model calls, cost per success. For `c1_reflect`: checks,
rejections, retries; **fix** = first answer fails the task's `number_match` and the final answer passes;
**harm** = first answer passes and the final fails; checker **false accept** = verdict ok but the
answer fails; **false reject** = verdict not ok but the first answer passes.

**H1 and decision rule** (benefit tasks, `c1_reflect` minus `c1_baseline`):

* **Supported:** mean difference >= +0.10, positive in at least 4 of the 5 repeat pairs (paired by
  repeat index), and harm count < fix count.
* **Not supported:** mean difference < +0.05 (including negative).
* **Inconclusive:** anything else.
* **Harm check (reported separately):** on the `easy` family, `c1_reflect` should not be more than
  0.05 below `c1_baseline`.

**Possible negative results** (to be reported as they occur): the checker accepts wrong answers (no
gain); the checker rejects right answers and the retry breaks them (harm); the retry repeats the same
mistake; cost roughly doubles with no gain; a gain on one family only; the baseline lands outside the
band (not run as designed).

### C1 trial record and freeze (added after the trials; the text above is unchanged)

Trial reports (baseline only, 1 repeat, 2026-11-19; not results):
`evals/reports/trials/exp_c1_trial_level1_baseline.json` and `..._level2_baseline.json`.

* **Level 1:** 18 of 18 benefit tasks passed (and 6 of 6 `easy`), above the band. Following the rule,
  the knob moved once, to level 2.
* **Level 2:** 15 of 18 benefit tasks passed (0.83), still above the band. Following the rule, level 2
  is frozen anyway; no further move is allowed. The three failures were read one by one and are model
  errors, not scoring bugs: `deep-06` converted ovo to sett by dividing instead of multiplying;
  `constraint-03` chose an item with no stock in the named warehouse; `constraint-02` gave the right
  item and amount but no `ANSWER:` line (a format failure, scored the same for every system).
  No reply was found that the scorers misjudged, so no task or scorer was changed.
* **Frozen:** `exp-c1-reflect` version `1-level2`, 24 tasks, content hash
  `5b6967ea267d2b476938483bb5caafc3b0db63d5e3a446ceeda6490be9485291` (`experiments/FROZEN.json`,
  key `v0.2`; file `evals/exp_c1_reflect/taskset.yaml`).
* **Consequence, stated before the full run:** if the full-run baseline also stays above 0.7 on the
  benefit tasks, C1 is reported as **not run as designed** (ceiling not avoided) and H1 is not decided,
  as pre-registered. The trials used 120,427 tokens (0.1096 CNY at list price).

## C2 (R2): when does the planner's parallelism save time?

**Question.** In v0.1.0 the planner took about twice as long as the single agent per fanout task, with
in-process tools that answer instantly (`docs/EXPERIMENTS.md`, supplementary timing). When each lookup
takes a while, the planner's workers can wait in parallel while the single agent waits one call after
another. From what tool latency, if any, is the planner faster, and what changes when tools fail
transiently? C2 supplements experiment A; it does not replace it.

**Systems** (same model and settings as C1):

* `c2_single`: one `Agent` with all C2 tools, `max_steps=12`. It runs its tool calls one after another.
* `c2_planned`: planner + research and compute workers + synthesiser, built like A's planner; the
  scheduler runs up to 4 workers in parallel (its default).

**Tools** (`experiments/c2_tasks.py`). `get_units(depot, product)` waits a fixed latency with a real
`asyncio.sleep` before answering; `calculate` has no delay. The latency is **simulated** (a sleep, not a
real service), so C2 speaks about the scheduling design, not about any real tool. Transient faults:
a lookup can fail with "depot service busy, try again"; the decision is a hash of (seed, tool,
normalised arguments, how many times this exact call was made before in the job), so both systems meet
the same faults whatever order they call in; one call fails at most twice in a row. A fault reaches the
model as a tool error in both systems (single agent and planner workers are both `Agent`s); the model
has to call again.

**Cells** (fixed): latency 1 s, 4 s and 10 s without faults; latency 4 s with fault rate 0.3.

**Task set `exp-c2-latency`**, 8 tasks on fictional data: units of a product at 3 or 4 depots, summed.
Scored with `number_match` (`which: marked`, tolerance 0.01) and `tool_called get_units` at least once
per depot. There is no difficulty knob; this experiment is about time, not pass rate.

**Trial.** Both systems, all 8 tasks, 1 repeat, in the cells `l1` and `l4f0.3`; caps 0.5 CNY and 0.2M
tokens. It checks wiring, the fault path and scoring; every reply is read. Not results. Scoring fixes
follow the C1 rule (task set only, regression test, documented; never a library scorer).

**Full run.** 3 repeats, every cell, both systems; one task at a time (concurrency 1) so each case's
wall-clock time is clean; chunks of one (repeat, system, cell) in a fixed alternating order within
each repeat; caps 2 CNY and 1M tokens. Wall time per case is the eval runner's own measurement
(`timing_ms` in each report), including failed runs.

**Metrics.** Per system and cell: median and mean wall time per task (24 runs), per-repeat mean, pass
rate, model calls, tokens and cost.

**H2 and decision rule.** In a fault-free cell, the planner **wins** if its median wall time per task
is below 0.9 times the single agent's **and** its per-repeat mean is lower in at least 2 of the 3
repeats. H2 is **supported** if the planner wins in at least one fault-free cell (the smallest such
latency is reported) and **not supported** if it wins in none.
**Secondary (descriptive, no decision):** at 4 s, how faults change both systems' times, pass rates
and model calls. If a system passes fewer than 80% of runs in a cell, its time there is reported with
that caveat.

**Possible negative results** (reported as they occur): the planner is never faster in the tested
range; it is faster only at 10 s; faults remove its advantage; its pass rate is lower than the single
agent's.
