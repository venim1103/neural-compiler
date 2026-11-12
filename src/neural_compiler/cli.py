"""Command-line entry point for neural-compiler."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from neural_compiler import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neural-compiler")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--version", action="store_true", help="Print package version")
    action.add_argument("--run-python", type=Path, metavar="FILE", help="Run a Python file in the P0 namespace executor")
    args = parser.parse_args(argv)
    if args.version:
        print(json.dumps({"name": "neural-compiler", "version": __version__}))
    elif args.run_python:
        from neural_compiler.bridge import run_python

        result = run_python(args.run_python.read_text(encoding="utf-8"))
        print(json.dumps(asdict(result)))
        return 0 if result.status == "ok" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
