"""Command-line entry point for neural-compiler."""

from __future__ import annotations

import argparse
import json
from neural_compiler import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neural-compiler")
    parser.add_argument("--version", action="store_true", help="Print package version")
    args = parser.parse_args(argv)
    if args.version:
        print(json.dumps({"name": "neural-compiler", "version": __version__}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
