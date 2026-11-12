"""Standalone jailed harness for the research Wasm and x86 emulation paths."""

from __future__ import annotations

import sys
from pathlib import Path


def run_wat(code: str, fuel: int) -> tuple[int, str]:
    import wasmtime

    config = wasmtime.Config()
    config.consume_fuel = True
    config.parallel_compilation = False
    # Wasmtime's large default virtual reservations conflict with RLIMIT_AS.
    config.memory_reservation = 0
    config.memory_guard_size = 0
    engine = wasmtime.Engine(config)
    try:
        module = wasmtime.Module(engine, code)
    except wasmtime.WasmtimeError as exc:
        return 2, str(exc)
    store = wasmtime.Store(engine)
    store.set_fuel(fuel)
    store.set_limits(memory_size=64 * 1024 * 1024, table_elements=100_000, instances=1)
    try:
        instance = wasmtime.Instance(store, module, [])
        solve = instance.exports(store).get("solve")
        if not isinstance(solve, wasmtime.Func):
            return 1, "Module must export a function named solve"
        signature = solve.type(store)
        if signature.params or signature.results != [wasmtime.ValType.i64()]:
            return 1, "solve must take no arguments and return one i64"
        return 0, str(solve(store))
    except wasmtime.Trap as exc:
        return (3 if exc.trap_code == wasmtime.TrapCode.OUT_OF_FUEL else 1), str(exc)
    except wasmtime.WasmtimeError as exc:
        return 1, str(exc)


def run_hex(code: str, max_insns: int) -> tuple[int, str]:
    from unicorn import UC_ARCH_X86, UC_MODE_64, Uc, UcError
    from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RIP, UC_X86_REG_RSP

    blob = bytes.fromhex(code)
    base, stack, ret = 0x100000, 0x800000, 0x300000
    emulator = Uc(UC_ARCH_X86, UC_MODE_64)
    emulator.ctl_set_tcg_buffer_size(32 * 1024 * 1024)
    emulator.mem_map(base, 0x100000)
    emulator.mem_map(stack - 0x100000, 0x200000)
    emulator.mem_map(ret, 0x1000)
    emulator.mem_write(base, blob)
    rsp = stack - 0x1000
    emulator.mem_write(rsp, ret.to_bytes(8, "little"))
    emulator.reg_write(UC_X86_REG_RSP, rsp)
    try:
        emulator.emu_start(base, ret, count=max_insns)
    except UcError as exc:
        return 1, str(exc)
    if emulator.reg_read(UC_X86_REG_RIP) != ret or emulator.reg_read(UC_X86_REG_RSP) != rsp + 8:
        return 3, "did not return before the instruction limit"
    value = emulator.reg_read(UC_X86_REG_RAX)
    return 0, str(value - (1 << 64) if value >> 63 else value)


if __name__ == "__main__":
    language, source, budget = sys.argv[1:]
    code = Path(source).read_text(encoding="utf-8")
    status, output = run_wat(code, int(budget)) if language == "wat" else run_hex(code, int(budget))
    print(output, file=sys.stdout if status == 0 else sys.stderr)
    raise SystemExit(status)
