"""Command-line entry point for neural-compiler."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from neural_compiler import __version__


def _read_program(path: Path) -> str:
    from neural_compiler.bridge.execution import MAX_CODE_BYTES

    with path.open("rb") as source:
        raw = source.read(MAX_CODE_BYTES + 1)
    if len(raw) > MAX_CODE_BYTES:
        raise ValueError(f"Program exceeds {MAX_CODE_BYTES} UTF-8 bytes")
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neural-compiler")
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Execution wall-clock timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--compiler",
        choices=("gcc", "tcc"),
        default="gcc",
        help="C compiler inside the jail for --run-c (default: gcc)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--version", action="store_true", help="Print package version")
    action.add_argument(
        "--run-python",
        type=Path,
        metavar="FILE",
        help="Run a Python file in the P0 namespace executor",
    )
    action.add_argument(
        "--run-c",
        type=Path,
        metavar="FILE",
        help="Compile and run a strict C99 file in the P1 jail executor",
    )
    action.add_argument(
        "--run-asm",
        type=Path,
        metavar="FILE",
        help="Assemble and run an x86-64 assembly file (`long long solve(void)`) in the P1 jail executor",
    )
    action.add_argument(
        "--check-imports",
        type=Path,
        metavar="FILE",
        help="Evaluate static Python AST imports against the candidate math/stdlib allowlist without executing",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(json.dumps({"name": "neural-compiler", "version": __version__}))
        return 0
    if args.check_imports:
        from neural_compiler.bridge import evaluate_import_policy

        report = evaluate_import_policy(_read_program(args.check_imports))
        print(json.dumps(asdict(report)))
        return 0 if report.compliant else 1
    if args.run_python or args.run_c or args.run_asm:
        from neural_compiler.bridge import run_asm, run_c, run_python

        if args.run_python:
            result = run_python(_read_program(args.run_python), timeout=args.timeout)
        elif args.run_c:
            result = run_c(
                _read_program(args.run_c),
                timeout=args.timeout,
                compiler=args.compiler,
            )
        else:
            result = run_asm(_read_program(args.run_asm), timeout=args.timeout)
        print(json.dumps(asdict(result)))
        return 0 if result.status == "ok" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
