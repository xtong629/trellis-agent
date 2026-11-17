# Experiments: design and change record

Two controlled comparisons (code in `experiments/`, task sets in `evals/exp_a_planner` and
`evals/exp_b_memory`, hashes in `experiments/FROZEN.json`). Design and families: see
`experiments/README.md`.

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

## Rules after the freeze

* The v2 task sets and scoring rules are not changed because of any pilot or full-run result.
* A scoring bug found later is reported first and fixed only after explicit agreement, with a new
  hash recorded in `FROZEN.json` and a full re-run.
