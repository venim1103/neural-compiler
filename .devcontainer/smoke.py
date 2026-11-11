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
    options = parser.parse_args()
    check_certificates()
    results = check_neurovm_pipeline()
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
