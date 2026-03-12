from __future__ import annotations

import argparse


def positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {raw_value!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"Value must be >= 1, got: {value}")
    return value


def non_negative_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {raw_value!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"Value must be >= 0, got: {value}")
    return value


def bounded_int(raw_value: str, *, minimum: int, maximum: int) -> int:
    value = positive_int(raw_value)
    if value < minimum or value > maximum:
        raise argparse.ArgumentTypeError(
            f"Value must be between {minimum} and {maximum}, got: {value}"
        )
    return value


def search_results_int(raw_value: str) -> int:
    return bounded_int(raw_value, minimum=1, maximum=20)
