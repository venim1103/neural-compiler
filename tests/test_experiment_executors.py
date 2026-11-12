"""Checks for the execution paths used by the recorded research experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from neural_compiler.bridge import run_python

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experiment_sandbox", ROOT / "docs/review/experiments/sandbox.py")
assert SPEC is not None and SPEC.loader is not None
sandbox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sandbox)


@pytest.mark.parametrize("code", ["b82a000000c3xyz", "b82a000000c", "", "hello"])
def test_invalid_hex_is_not_silently_rewritten(code: str) -> None:
    status, _ = sandbox.run_hex(code)
    assert status == "compile_error"


def test_hex_accepts_byte_separators(jail) -> None:
    assert sandbox.run_hex("b8 2a 00 00 00\nc3") == ("ok", "42")


@pytest.fixture
def jail():
    result = run_python("print(0)")
    if result.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {result.stderr}")
    assert result.status == "ok", result


def test_c_compiles_and_executes_inside_jail(jail) -> None:
    assert sandbox.run_c('#include <stdio.h>\nint main(void) { printf("42\\n"); }') == ("ok", "42")


def test_asm_compiles_and_executes_inside_jail(jail) -> None:
    assert sandbox.run_asm(".intel_syntax noprefix\n.global solve\nsolve:\nmov eax, 42\nret") == ("ok", "42")


@pytest.mark.parametrize("code,status", [
    ('(module (func (export "solve") (result i64) (i64.const 42)))', "ok"),
    ("not wat", "compile_error"),
    ('(module (func (export "solve") (result i32) (i32.const 42)))', "runtime_error"),
    ('(module (func (export "solve") (result i64) (loop $l (br $l)) (i64.const 0)))', "timeout"),
    ('(module (memory 1025) (func (export "solve") (result i64) (i64.const 42)))', "runtime_error"),
])
def test_wat_isolated_outcomes(jail, code: str, status: str) -> None:
    actual, output = sandbox.run_wat(code, fuel=1000)
    assert actual == status, output
    if status == "ok":
        assert output == "42"


def test_hex_instruction_exhaustion_is_not_success(jail) -> None:
    # pop rax restores RSP, but does not actually return to the sentinel address.
    status, _ = sandbox.run_hex("58 eb fe", max_insns=10)
    assert status == "timeout"


def test_c_rejects_non_c99_extensions(jail) -> None:
    status, _ = sandbox.run_c("int main(void) { int f(void) { return 42; } return f(); }")
    assert status == "compile_error"


@pytest.mark.parametrize("language", ["c", "asm"])
def test_compiler_cannot_read_host_file(jail, tmp_path: Path, language: str) -> None:
    host_file = tmp_path / "host-only"
    host_file.write_text("HOST_ONLY_TEST_SENTINEL", encoding="utf-8")
    if language == "c":
        code = f'#include "{host_file}"\nint main(void) {{ return 0; }}'
    else:
        code = f'.incbin "{host_file}"\n.global solve\nsolve: ret'
    status, output = sandbox.EXECUTORS[language](code)
    assert status == "compile_error"
    assert "HOST_ONLY_TEST_SENTINEL" not in output
