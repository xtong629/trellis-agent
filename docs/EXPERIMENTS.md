# Experiments

Two controlled comparisons on one model, with fictional data: does a planner help a tool-using agent
(A), and does tiered memory help a conversational agent compared with a sliding window (B). This
page records what was run, the numbers, and what they do and do not show. Code is in `experiments/`,
task sets in `evals/exp_a_planner` and `evals/exp_b_memory`, reports in `evals/reports/`.

## Task sets and why they are built this way

All data is fictional (item prices, stock, populations, exchange rates, allergens, deadlines,
refund numbers), so nothing can be answered from world knowledge. Expected answers are computed from
that data by a generator (`python -m experiments.build`). Scoring uses deterministic rules only (no LLM
judge). Every set has families where the system under test *should* help and families where it
*should not*, so a result can also show no effect. The hypothesis for each family is stored in each
task's `metadata.why`.

| Set | Family (tasks) | Expected | Reason |
| --- | --- | --- | --- |
| A, 30 tasks | serial (8) | benefit | price, then exchange rate, then multiply: dependent steps |
| | fanout (8) | benefit | independent lookups followed by a sum |
| | flaky (6) | benefit | the exchange-rate service is busy on its first call (retry) |
| | single (8) | no benefit | one lookup |
| B, 20 scripts | allergy (6) | benefit | an early fact asked about after many unrelated turns |
| | deadline (5) | benefit | an early fact; three scripts later correct it |
| | refund-zh (3) | benefit | the same, in Chinese |
| | fits (3) | no benefit | the whole conversation fits in the window |
| | irrelevant (3) | no benefit | the last question needs nothing remembered |

B scripts are multi-turn: turns are separated by a `---` line in `input` and only the last reply is
scored. Task set hashes (v2): A `af2621c4ee55728b97574a46e4b1d57fe36159d45f1349bff93f2be1a0725a4b`, B `1c1a372d823ceace4a835616cdf64ca33c3ac08c00d2c99b63d21bd25ba9294f`.

## v1 -> v2 change record

v1 hashes: A `25c438fe84a9d84a0eb8b6b7384c197d9d6a2f6f6305347e9a0d3e459bb4c06c`,
B `76c0056431fccb306f9bd46b171c170baba315ef0e9d4c1424fe7c8a516ac1e1` (full values: `FROZEN.json`).
v2 hashes are in `FROZEN.json` under `v2`.

A short pilot against the real API (19 tasks, one repeat) showed three problems. All three are defects
in how the tasks were scored or sized, found by reading individual replies; the fixes below are
justified by what each rule should mean, not by any outcome, and no pass rate is quoted here.

1. **`number_match` with `which: last` scored the wrong number.** It took the last number anywhere in
   the reply, so a reply that gave the right answer followed by its working ("costs 21 krell
   (14 zorin x 1.5)") was judged by the 1.5.
   *Fix:* the scorer gained `which: marked`: the first number on the last line that starts with
   `ANSWER:` is the answer; working elsewhere cannot matter, and a missing or wrong marked line fails
   with a reason naming the marker. All numeric tasks now end their prompt with "End your reply with a
   line of the form 'ANSWER: <number>'." and use `which: marked`.
   *Why this way:* the answer is defined by an explicit marker the task asks for, not by a guess about
   which number the model meant. A reply that ignores the format instruction fails; that is a
   format-following failure and is the same for every system compared.
   Regression tests use the reply shapes that were mis-scored.
2. **`contains` was case-sensitive.** "Zandal nut" did not match "zandal nut".
   *Fix:* `contains` and `not_contains` ignore case by default (`casefold: false` restores strictness).
   *Why:* none of these tasks is about letter case; capitalisation at the start of a sentence says
   nothing about whether a fact was remembered. `not_contains` follows the same default so a forbidden
   date cannot hide behind capital letters. No task needs case-sensitive matching.
3. **The experiment B window was not sized by any rule.** A window of 100 tokens could not hold even
   the complete `fits` conversation, so the "should not benefit" group was not a fair control.
   *Fix:* the window is derived from the task set alone (`experiments/window.py`):
   the window must hold the complete conversation of every `fits` script and be too small for the
   complete conversation of every other script.
   A conversation is measured with the library's own token estimate (chars / 4 per message) as
   `system prompt + every user turn + 40 tokens per reply`. The 40 is the size replies are asked to
   stay within: the B system prompt now says "Reply in at most 20 words" (about 27 tokens, rounded up).
   The window is the largest `fits` conversation rounded up to a multiple of 10.
   *Derivation for v2:* `fits` conversations measure 143, 144 and 145 tokens, so the window is 150;
   every other script measures at least 359. `derive_window` raises if the two groups cannot be
   separated, and a test checks that the value in use equals the derived one.
   *The value is final:* it is not adjusted after any result.

Both systems in B use the same window and system prompt.

## Method

* Model: `glm-5.3-flash` through an OpenAI-compatible API, `reasoning_effort=low` for every call.
* Repeats: 5 per task and system. Success rate is the mean and the sample standard deviation (n-1)
  of the per-repeat pass rate over the 5 repeats.
* Concurrency: 2 tasks at a time; the planner's scheduler may run up to 4 workers in parallel
  (theoretical maximum 8 requests in flight for the planner, 2 for the others). The highest observed
  was 6 (`evals/reports/exp_full_http.json`) and 2 for the full-history run
  (`evals/reports/exp_b_full_http.json`). The limits are recorded in each report's `config.concurrency`.
* Price table: `evals/prices.glm.yaml` (file hash `dad4039c29e0`, in each report's `config`), CNY, list
  prices. A token package is used in practice, so every cost below is a **reference value at list
  price, not actual spend**.
* Budget cap 5 CNY per run (the main run and the full-history run each); neither came close.
* Date: 2026-11-18. Same system prompt for all B systems; B window of 150 tokens (rule above).
* Systems: A `a_single` (one agent with all tools) vs `a_planned` (planner, research and compute
  workers, synthesiser). B `b_window` (sliding window only), `b_memory` (the same window plus tiered
  memory with an LLM fact extractor), `b_full` (reference: the whole conversation every turn).
* Run order: the four main systems were run first; `b_full` was added afterwards as an upper-bound
  reference, with the task set and scoring unchanged.
* HTTP: 4,251 requests for the main run and 855 for the reference run, all status 200, no 429, no
  retries (`exp_full_http.json`, `exp_b_full_http.json`).

## Results

All numbers come from the report named in the table (`evals/reports/`) and can be regenerated with
`python -m experiments.results_table`. Task sets v2, 5 repeats each.

| system | report | tasks x repeats | all | should benefit | should not benefit | cost (CNY, list price) | cost per success | tokens | model calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single agent | `exp_a_single.json` | 30 x 5 | 0.987±0.018 | 0.982±0.025 | 1.000±0.000 | 0.2331 | 0.00158 | 263,405 | 463 |
| planner | `exp_a_planned.json` | 30 x 5 | 0.993±0.015 | 0.991±0.020 | 1.000±0.000 | 0.4877 | 0.00327 | 506,477 | 1,223 |
| sliding window | `exp_b_window.json` | 20 x 5 | 0.400±0.035 | 0.171±0.039 | 0.933±0.091 | 0.1515 | 0.00379 | 127,037 | 855 |
| tiered memory | `exp_b_memory.json` | 20 x 5 | 0.980±0.027 | 0.986±0.032 | 0.967±0.075 | 0.2697 | 0.00275 | 244,080 | 1,710 |
| full history (reference) | `exp_b_full.json` | 20 x 5 | 0.980±0.027 | 1.000±0.000 | 0.933±0.091 | 0.2062 | 0.00210 | 196,582 | 855 |

Experiment A by family (success rate, mean±sd over 5 repeats):

| family (tasks) | single agent | planner |
| --- | --- | --- |
| serial (8) | 1.000±0.000 | 1.000±0.000 |
| fanout (8) | 0.950±0.068 | 1.000±0.000 |
| flaky (6) | 1.000±0.000 | 0.967±0.075 |
| single (8) | 1.000±0.000 | 1.000±0.000 |

Experiment B by family:

| family (tasks) | sliding window | tiered memory | full history (reference) |
| --- | --- | --- | --- |
| allergy (6) | 0.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| deadline (5) | 0.480±0.110 | 1.000±0.000 | 1.000±0.000 |
| refund-zh (3) | 0.000±0.000 | 0.933±0.149 | 1.000±0.000 |
| fits (3) | 0.867±0.183 | 0.933±0.149 | 0.867±0.183 |
| irrelevant (3) | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |

### What the numbers show

* **A, planner vs single agent.** On this task set and this model the planner did not change accuracy
  measurably: 0.993±0.015 against 0.987±0.018 (150 runs each; 3 failed runs in total, 2 for the
  single agent on fanout-03 and 1 for the planner on flaky-05). It cost about 2.1 times as much
  (0.4877 vs 0.2331 CNY at list price), used about 1.9 times the tokens (506,477 vs 263,405) and
  2.6 times the model calls (1,223 vs 463). The single agent was already at 0.987, so there was little
  room to improve (ceiling effect).
* **B, memory vs sliding window.** With the 150-token window, the window system recalled 0.171±0.039
  of the should-benefit scripts and tiered memory 0.986±0.032 (100 runs each in the full table, 70
  of them should-benefit). Full history, the reference, scored 1.000±0.000 on those. Memory came
  within one failed run (refund-zh, 1 of 15) of full history on every family.
* **Cost of memory in B.** Memory used about 1.8 times the window system's cost (0.2697 vs 0.1515 CNY)
  because it makes an extra model call per turn to extract facts (1,710 vs 855 calls). On these
  short scripts (3 to 13 turns) full history was cheaper than memory (0.2062 vs 0.2697 CNY, 196,582
  vs 244,080 tokens). Memory's advantage over full history would have to come from much longer
  conversations, which were not tested.
* **No-benefit groups.** Window, memory and full history are within noise of each other on `fits` and
  `irrelevant` (30 runs in total per system; the standard deviations are 0.07 to 0.09).

## Limitations and conclusions that cannot be drawn

* **Ceiling effect in A.** The single agent solved about 99% of the tasks, so this set cannot show a
  planner improving accuracy; it only shows the planner is not needed for tasks of this difficulty
  and costs more. Harder tasks might behave differently.
* **The B gap depends on the window size.** The window (150 tokens) was derived by a rule so that
  `fits` conversations fit and all others overflow. The size of the window-versus-memory gap follows
  directly from that choice. It says memory preserves facts that fall out of a small window; it does
  not measure how much better memory is than windows in general.
* **Extractor coupling (disclosure).** The memory system in the runs used the LLM extractor
  (`LLMExtractor`), not the rule-based `RuleExtractor`. The scripts were nonetheless written in
  fixed sentence shapes. Checked offline (no model) against `RuleExtractor`: of 23 fact statements
  in the 20 scripts (12 allergy statements, 5 initial deadlines, 3 deadline corrections, 3 Chinese
  refund statements) it recognised 8: all 5 initial deadlines ("The report deadline is ...") and the 3
  Chinese ones (with an over-long key), and none of the 12 allergy statements or the 3 corrections. So
  the deadline and Chinese sentences have the shape the rule extractor expects; the allergy and
  correction sentences do not. This did not affect the reported runs, but the scripts are not
  independent of the library's own sentence patterns, and the offline mock memory path is coupled to
  the script wording by construction.
* **Lexical retrieval is favoured by construction.** B's final questions share a word with the stored
  fact (for example "allergic", "deadline", the Chinese term for refund number). Retrieval is lexical
  (full-text search); paraphrased questions were not tested.
* **Sample size.** 30 (A) and 20 (B) tasks, 5 repeats each. A group of 3 tasks and 5 repeats is 15
  runs, so one failed run moves its rate by 0.067; the standard deviation is over only 5 repeats.
  No significance tests were run, and the repeats are not independent samples of the task space.
* **Fictional data, one model, one effort level.** `glm-5.3-flash` at `reasoning_effort=low` only.
  `low` returns no reasoning content and its effect on quality was not measured. Results may not
  carry over to other models or settings. Real-world tasks were not tested.
* **Unfamiliar words.** On the `fits` family, full history failed twice on one script (fits-02), where
  the model answered a similar real ingredient instead of the fictional allergen; that group cannot
  separate memory from reading errors.
* **Scoring format.** Numeric answers must end with an `ANSWER:` line; a reply that ignores the
  format fails. Two failures of the single agent on fanout-03 are replies that reported being unable
  to retrieve the stock counts.
* **Costs are reference values.** CNY list prices; a free token package was used, so no real money was
  spent. Reported cost per success is cost divided by passed runs.
* **Wall-clock time is not reported.** The reports were produced without per-case timing, so there is
  no wall-clock comparison by group (for example fanout) and no claim is made about latency. Only the
  total chunk wall time of the main run is recorded (4,707 s, `exp_full_http.json`).
* **The pilot informed the v2 fixes.** v2 changed scoring and window sizing after a 19-task pilot; the
  reasons are recorded above and no task or scoring change was made after the v2 freeze.

## Rules after the freeze

* The v2 task sets and scoring rules are not changed because of any pilot or full-run result.
* A scoring bug found later is reported first and fixed only after explicit agreement, with a new
  hash recorded in `FROZEN.json` and a full re-run.
