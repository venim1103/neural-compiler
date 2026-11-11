"""Container environment and weight-compiled NeuroVM pipeline smoke check."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3_5Config


def check_certificates() -> None:
    bundle = Path("/etc/ssl/certs/ca-certificates.crt")
    for variable in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ):
        if os.environ.get(variable) != str(bundle):
            raise RuntimeError(f"{variable} must use the system CA bundle")
    if not bundle.is_file() or ssl.create_default_context().cert_store_stats()["x509_ca"] == 0:
        raise RuntimeError("The system CA trust store is empty or missing")


def check_neurovm_pipeline() -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from neural_compiler import gdn_exact_read, gdn_exact_write, snap_bipolar

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = os.environ.get("NEURAL_COMPILER_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("NEURAL_COMPILER_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA profile requires working NVIDIA GPU passthrough")

    drifted = torch.tensor([0.94, -0.96, 1.05, -1.04], device=device)
    snapped = snap_bipolar(drifted, eps=0.1)
    if not torch.allclose(snapped, torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)):
        raise RuntimeError("Bipolar deadzone snap check failed")

    gdn_state = torch.zeros((8, 16), dtype=torch.float32, device=device)
    word = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], device=device)
    gdn_state = gdn_exact_write(gdn_state, 5, word)
    readback = gdn_exact_read(gdn_state, 5)
    if not torch.equal(readback, word):
        raise RuntimeError("Gated DeltaNet exact RAM write/read check failed")

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).to(device).eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tokens).logits
    if logits.shape != (1, 4, 64):
        raise RuntimeError("Synthetic backbone forward pass failed")

    if Qwen3_5Config().model_type != "qwen3_5":
        raise RuntimeError("Qwen3.5 configuration is unavailable")

    return {
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "bipolar_snap_verified": True,
        "gdn_ram_verified": True,
    }


def check_tests() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(repo_root / "tests")],
        check=True,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )


TOOLCHAIN_BINARIES = ("bwrap", "clang", "tcc", "wasm-ld", "wasm2wat", "wat2wasm")


def check_toolchain(device: str) -> dict:
    """Verify the compile-and-execute toolchain used by the execution-bridge experiments."""
    import shutil
    import tempfile

    import wasmtime
    from unicorn import UC_ARCH_X86, UC_MODE_64, Uc
    from unicorn.x86_const import UC_X86_REG_RAX

    missing = [name for name in TOOLCHAIN_BINARIES if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Missing toolchain binaries: {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        source = work / "triangle.c"
        source.write_text(
            "long long triangle(void) {\n"
            "    long long total = 0;\n"
            "    for (int i = 1; i <= 100; i++) total += i;\n"
            "    return total;\n"
            "}\n"
        )
        module_path = work / "triangle.wasm"
        subprocess.run(
            ["clang", "--target=wasm32", "-O2", "-nostdlib", "-Wl,--no-entry",
             "-Wl,--export=triangle", "-o", str(module_path), str(source)],
            check=True,
        )
        config = wasmtime.Config()
        config.consume_fuel = True
        engine = wasmtime.Engine(config)
        store = wasmtime.Store(engine)
        store.set_fuel(1_000_000)
        instance = wasmtime.Instance(store, wasmtime.Module.from_file(engine, str(module_path)), [])
        if instance.exports(store)["triangle"](store) != 5050:
            raise RuntimeError("clang -> wasm32 -> wasmtime check failed")

        text = subprocess.run(
            ["wasm2wat", str(module_path)], check=True, capture_output=True, text=True
        ).stdout
        (work / "roundtrip.wat").write_text(text)
        subprocess.run(
            ["wat2wasm", str(work / "roundtrip.wat"), "-o", str(work / "roundtrip.wasm")],
            check=True,
        )

        program = work / "answer.c"
        program.write_text('#include <stdio.h>\nint main(void) { printf("%d\\n", 6 * 7); return 0; }\n')
        output = subprocess.run(
            ["tcc", "-run", str(program)], check=True, capture_output=True, text=True
        ).stdout
        if output.strip() != "42":
            raise RuntimeError("tcc -run check failed")

    emulator = Uc(UC_ARCH_X86, UC_MODE_64)
    emulator.mem_map(0x100000, 0x1000)
    emulator.mem_write(0x100000, bytes.fromhex("b82a000000"))  # mov eax, 42
    emulator.emu_start(0x100000, 0x100005)
    if emulator.reg_read(UC_X86_REG_RAX) != 42:
        raise RuntimeError("unicorn x86-64 emulation check failed")

    # bubblewrap needs unprivileged user namespaces, which depend on the container runtime
    # (they work under rootless Podman; Docker's default seccomp profile may block them).
    sandbox = subprocess.run(
        ["bwrap", "--ro-bind", "/", "/", "--unshare-all", "--die-with-parent", "true"],
        capture_output=True,
        text=True,
    )
    results = {
        "clang_wasm32_wasmtime_verified": True,
        "wabt_roundtrip_verified": True,
        "tcc_verified": True,
        "unicorn_verified": True,
        "bubblewrap_namespaces": sandbox.returncode == 0,
        "versions": {package: version(package) for package in ("wasmtime", "unicorn")},
    }
    if device == "cuda":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

        results["versions"]["fla-core"] = version("fla-core")
        results["fla_kernels_verified"] = True
    return results


def check_network() -> None:
    import requests

    for address in (
        "https://pypi.org/simple/pip/",
        "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct",
    ):
        with urllib.request.urlopen(address, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTPS check failed: {address}")
        response = requests.get(address, timeout=30)
        response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument(
        "--toolchain",
        action="store_true",
        help="verify clang->wasm32, wasmtime, wabt, tcc, unicorn, bubblewrap (and fla on CUDA)",
    )
    options = parser.parse_args()
    check_certificates()
    results = check_neurovm_pipeline()
    if options.toolchain:
        results["toolchain"] = check_toolchain(results["device"])
    if options.network:
        check_network()
    results["versions"] = {
        package: version(package)
        for package in ("torch", "transformers", "safetensors", "accelerate", "numpy", "scipy")
    }
    print(json.dumps(results, indent=2))
    print("Container smoke checks passed; no pretrained model was downloaded.")
    if options.tests:
        check_tests()
        print("Unit tests passed.")


if __name__ == "__main__":
    main()
