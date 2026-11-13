"""Deterministic synthetic task-family generators for offline harness verification.

These parameterized generators produce synthetic unit fixtures for harness and
reporting checks; they are not accepted benchmark splits or accuracy evidence.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from fractions import Fraction

from .harness import (
    IntegerChecker,
    IntegerSequenceChecker,
    RationalChecker,
    Task,
)

_PRIMES_LARGE = (1000000007, 1000000009, 998244353, 1000000033)
_PRIMES_MEDIUM = (10007, 65537, 100003, 1000003)
_RECURRENCE_MODULI = (997, 1009, 10007, 65537)


def _sample_modular_exponentiation(
    rng: random.Random, case_id: str, family: str
) -> tuple[tuple[int, ...], Task]:
    modulus = rng.choice(_PRIMES_LARGE)
    base = rng.randrange(2, 500)
    exponent = rng.randrange(10**7, 10**9)
    expected = pow(base, exponent, modulus)
    task = Task(
        case_id=case_id,
        family=family,
        prompt=f"Compute {base} to the power {exponent}, modulo {modulus}.",
        checker=IntegerChecker(expected),
    )
    return (base, exponent, modulus), task


def _sample_binomial_coefficient(
    rng: random.Random, case_id: str, family: str
) -> tuple[tuple[int, ...], Task]:
    n = rng.randrange(24, 65)
    k = rng.randrange(6, n // 2 + 1)
    expected = math.comb(n, k)
    task = Task(
        case_id=case_id,
        family=family,
        prompt=f"Compute the binomial coefficient C({n}, {k}).",
        checker=IntegerChecker(expected),
    )
    return (n, k), task


def _sample_linear_congruence(
    rng: random.Random, case_id: str, family: str
) -> tuple[tuple[int, ...], Task]:
    modulus = rng.choice(_PRIMES_MEDIUM)
    multiplier = rng.randrange(2, modulus - 1)
    remainder = rng.randrange(1, modulus)
    expected = (remainder * pow(multiplier, -1, modulus)) % modulus
    task = Task(
        case_id=case_id,
        family=family,
        prompt=(
            f"Find the unique integer x in 0 <= x < {modulus} "
            f"such that ({multiplier} * x) mod {modulus} = {remainder}."
        ),
        checker=IntegerChecker(expected),
    )
    return (multiplier, remainder, modulus), task


def _sample_harmonic_partial_sum(
    rng: random.Random, case_id: str, family: str
) -> tuple[tuple[int, ...], Task]:
    start = rng.randrange(1, 12)
    length = rng.randrange(4, 14)
    end = start + length - 1
    expected = sum((Fraction(1, k) for k in range(start, end + 1)), Fraction(0))
    task = Task(
        case_id=case_id,
        family=family,
        prompt=(
            f"Compute the exact reduced fraction sum(1/k for k from {start} to {end}) "
            "as <numerator>/<denominator>."
        ),
        checker=RationalChecker(expected),
        answer_format="<numerator>/<denominator>",
    )
    return (start, end), task


def _sample_recurrence_prefix(
    rng: random.Random, case_id: str, family: str
) -> tuple[tuple[int, ...], Task]:
    modulus = rng.choice(_RECURRENCE_MODULI)
    x0 = rng.randrange(1, modulus)
    a = rng.randrange(2, 500)
    c = rng.randrange(1, 500)
    length = rng.randrange(5, 9)
    terms = [x0]
    for _ in range(length - 1):
        terms.append((a * terms[-1] + c) % modulus)
    expected = tuple(terms)
    task = Task(
        case_id=case_id,
        family=family,
        prompt=(
            f"Let x_0 = {x0} and x_(t+1) = ({a} * x_t + {c}) mod {modulus}. "
            f"List the first {length} terms x_0 through x_{length - 1} separated by commas."
        ),
        checker=IntegerSequenceChecker(expected),
        answer_format="<n1>, <n2>, ...",
    )
    return (x0, a, c, modulus, length), task


_GENERATORS: dict[
    str, Callable[[random.Random, str, str], tuple[tuple[int, ...], Task]]
] = {
    "modular_exponentiation": _sample_modular_exponentiation,
    "binomial_coefficient": _sample_binomial_coefficient,
    "linear_congruence": _sample_linear_congruence,
    "harmonic_partial_sum": _sample_harmonic_partial_sum,
    "recurrence_prefix": _sample_recurrence_prefix,
}

SYNTHETIC_FAMILIES: tuple[str, ...] = tuple(_GENERATORS)


def generate_family_tasks(family: str, count: int, seed: int = 0) -> tuple[Task, ...]:
    """Generate `count` distinct, deterministic tasks for one synthetic family."""
    if family not in _GENERATORS:
        raise ValueError(f"Unknown synthetic task family: {family}")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    sampler = _GENERATORS[family]
    seen: set[tuple[int, ...]] = set()
    tasks: list[Task] = []
    max_attempts = count * 25
    attempts = 0
    while len(tasks) < count:
        if attempts >= max_attempts:
            raise ValueError(
                f"Cannot sample {count} distinct tasks for family {family!r}"
            )
        attempts += 1
        case_id = f"{family}-s{seed}-{len(tasks)}"
        params, task = sampler(rng, case_id, family)
        if params in seen:
            continue
        seen.add(params)
        tasks.append(task)
    return tuple(tasks)


def generate_matched_suite(
    families: Sequence[str] | None = None,
    per_family: int = 3,
    seed: int = 0,
) -> tuple[Task, ...]:
    """Generate a balanced deterministic suite across multiple synthetic families."""
    selected = SYNTHETIC_FAMILIES if families is None else tuple(families)
    if not selected:
        raise ValueError("families must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("families must be unique")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    suite: list[Task] = []
    for index, family in enumerate(selected):
        suite.extend(
            generate_family_tasks(family, per_family, seed=seed + index * 1009)
        )
    return tuple(suite)
