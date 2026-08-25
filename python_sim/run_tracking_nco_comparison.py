"""Command-line float/30-bit/43-bit tracking Carrier NCO comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from antenna_diversity.analysis.tracking_nco import (
    compare_tracking_ncos,
    save_tracking_nco_comparison,
)
from antenna_diversity.config import default_config


def _word_bits(value: str) -> tuple[int, ...]:
    try:
        bits = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "word bits must be comma-separated integers"
        ) from error
    if not bits:
        raise argparse.ArgumentTypeError(
            "at least one word length is required"
        )
    if len(set(bits)) != len(bits) or any(
        not 2 <= item <= 62 for item in bits
    ):
        raise argparse.ArgumentTypeError(
            "word bits must be unique integers in [2, 62]"
        )
    return bits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare float and fixed-point tracking Carrier NCOs."
        )
    )
    parser.add_argument("--clock-hz", type=float, default=6.6e6)
    parser.add_argument(
        "--word-bits", type=_word_bits, default=(30, 43)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tracking_nco"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = default_config()
    cfg.plot.enable = False
    comparison = compare_tracking_ncos(
        cfg,
        clock_hz=args.clock_hz,
        fixed_word_bits=args.word_bits,
    )
    paths = save_tracking_nco_comparison(
        comparison,
        args.output_dir,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
