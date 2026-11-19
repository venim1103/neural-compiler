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
    parser.add_argument(
        "--wat-fuel",
        type=int,
        metavar="COUNT",
        help="Override the API fuel limit for --run-wat",
    )
    parser.add_argument(
        "--hex-max-insns",
        type=int,
        metavar="COUNT",
        help="Override the API instruction limit for --run-hex",
    )
    parser.add_argument(
        "--eval-tasks",
        type=Path,
        metavar="FILE",
        help="Check exact task references with --eval-report or --serving-report",
    )
    parser.add_argument(
        "--eval-development-tasks",
        type=Path,
        metavar="FILE",
        help="Report declared development-task overlaps alongside --eval-tasks",
    )
    parser.add_argument(
        "--check-serving-logits",
        action="store_true",
        help="Check recorded CPU logits against full reprefill with --serving-report",
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
        "--run-wat",
        type=Path,
        metavar="FILE",
        help="Run WAT exporting solve() -> i64 in a jailed Wasmtime worker",
    )
    action.add_argument(
        "--run-hex",
        type=Path,
        metavar="FILE",
        help="Emulate x86-64 machine-code hex in a jailed Unicorn worker",
    )
    action.add_argument(
        "--check-imports",
        type=Path,
        metavar="FILE",
        help="Evaluate static Python AST imports against the candidate math/stdlib allowlist without executing",
    )
    action.add_argument(
        "--eval-report",
        type=Path,
        metavar="FILE",
        help="Replay a matched report from saved case evidence without model or program execution",
    )
    action.add_argument(
        "--serving-report",
        type=Path,
        metavar="FILE",
        help="Replay saved serving-comparison observations without inference or execution",
    )
    args = parser.parse_args(argv)
    if args.eval_tasks is not None and not (args.eval_report or args.serving_report):
        parser.error("--eval-tasks requires --eval-report or --serving-report")
    if args.check_serving_logits and args.serving_report is None:
        parser.error("--check-serving-logits requires --serving-report")
    if args.eval_development_tasks is not None and args.eval_tasks is None:
        parser.error("--eval-development-tasks requires --eval-tasks")
    for flag, limit, source, language in (
        ("--wat-fuel", args.wat_fuel, args.run_wat, "wat"),
        ("--hex-max-insns", args.hex_max_insns, args.run_hex, "hex"),
    ):
        if limit is not None:
            if source is None:
                parser.error(f"{flag} requires --run-{language}")
            from neural_compiler.bridge.native import _validate_runtime_budget

            _validate_runtime_budget(language, limit, flag)
    if args.version:
        print(json.dumps({"name": "neural-compiler", "version": __version__}))
        return 0
    if args.eval_report or args.serving_report:
        from neural_compiler.eval import (
            ServingCaseResult,
            inspect_task_overlap,
            load_case_results,
            load_serving_results,
            load_tasks,
            matched_report,
            summarize_serving,
            validate_references,
            validate_serving_logits,
        )

        serving_results: tuple[ServingCaseResult, ...] | None = None
        report_path = args.eval_report or args.serving_report
        with report_path.open(encoding="utf-8") as source:
            if args.eval_report:
                results = load_case_results(source)
            else:
                serving_results = load_serving_results(
                    source, include_logits=args.check_serving_logits
                )
                results = tuple(row.case for row in serving_results)
        reference_checks: dict[str, int] | None = None
        task_overlap: dict[str, object] | None = None
        if args.eval_tasks is not None:
            with args.eval_tasks.open(encoding="utf-8") as source:
                tasks = load_tasks(source)
            reference_checks = validate_references(tasks, results)
            if args.eval_development_tasks is not None:
                with args.eval_development_tasks.open(encoding="utf-8") as source:
                    development = load_tasks(source)
                task_overlap = inspect_task_overlap(development, tasks)
        if serving_results is None:
            report = matched_report(results)
        else:
            logit_checks = (
                validate_serving_logits(serving_results)
                if args.check_serving_logits
                else None
            )
            report = summarize_serving(serving_results)
            if logit_checks is not None:
                report["logit_validation"] = logit_checks
        if reference_checks is not None:
            report["reference_validation"] = reference_checks
        if task_overlap is not None:
            report["task_overlap"] = task_overlap
        print(json.dumps(report, allow_nan=False))
        return 0
    if args.check_imports:
        from neural_compiler.bridge import evaluate_import_policy

        report = evaluate_import_policy(_read_program(args.check_imports))
        print(json.dumps(asdict(report)))
        return 0 if report.compliant else 1
    if args.run_python or args.run_c or args.run_asm or args.run_wat or args.run_hex:
        from neural_compiler.bridge import run_asm, run_c, run_hex, run_python, run_wat

        if args.run_python:
            result = run_python(_read_program(args.run_python), timeout=args.timeout)
        elif args.run_c:
            result = run_c(
                _read_program(args.run_c),
                timeout=args.timeout,
                compiler=args.compiler,
            )
        elif args.run_asm:
            result = run_asm(_read_program(args.run_asm), timeout=args.timeout)
        elif args.run_wat:
            result = run_wat(
                _read_program(args.run_wat),
                timeout=args.timeout,
                **({"fuel": args.wat_fuel} if args.wat_fuel is not None else {}),
            )
        else:
            result = run_hex(
                _read_program(args.run_hex),
                timeout=args.timeout,
                **(
                    {"max_insns": args.hex_max_insns}
                    if args.hex_max_insns is not None
                    else {}
                ),
            )
        print(json.dumps(asdict(result)))
        return 0 if result.status == "ok" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
