"""Agreement between two pass/fail graders, for calibrating an LLM judge against rules or labels.

Reports the 2x2 counts, the raw agreement rate, Cohen's kappa (agreement corrected for chance) and a
percentile bootstrap interval for kappa. Kappa is undefined (``None``) when the chance agreement is
1, for example when both graders give the same verdict to everything.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

Pair = tuple[bool, bool]  # (first grader passed, second grader passed)


def cohen_kappa(pairs: Sequence[Pair]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    first = sum(a for a, _ in pairs) / n
    second = sum(b for _, b in pairs) / n
    chance = first * second + (1 - first) * (1 - second)
    if chance >= 1.0:
        return None
    return (observed - chance) / (1 - chance)


@dataclass(frozen=True)
class Agreement:
    n: int
    both_pass: int
    both_fail: int
    first_only: int  # first grader passed, second failed
    second_only: int  # second grader passed, first failed
    agree_rate: float | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    ci_level: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def agreement(
    pairs: Sequence[Pair], *, bootstrap: int = 2000, seed: int = 0, level: float = 0.95
) -> Agreement:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    n = len(pairs)
    kappa = cohen_kappa(pairs)
    interval: tuple[float, float] | None = None
    if bootstrap > 0 and n > 0 and kappa is not None:
        rng = random.Random(seed)
        draws = [cohen_kappa([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(bootstrap)]
        values = sorted(k for k in draws if k is not None)
        if values:
            tail = (1 - level) / 2
            low = values[int(tail * (len(values) - 1))]
            high = values[int((1 - tail) * (len(values) - 1))]
            interval = (low, high)
    return Agreement(
        n=n,
        both_pass=sum(a and b for a, b in pairs),
        both_fail=sum(not a and not b for a, b in pairs),
        first_only=sum(a and not b for a, b in pairs),
        second_only=sum(b and not a for a, b in pairs),
        agree_rate=None if n == 0 else sum(a == b for a, b in pairs) / n,
        kappa=kappa,
        kappa_ci=interval,
        ci_level=level,
    )
