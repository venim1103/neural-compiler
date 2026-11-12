# Development Container (`neural-compiler`)

This setup adapts the multi-profile Dev Container and certificate-trust architecture from `llm-loop-fix`, `micro_classifier`, and `embedded-jev`. It provides hash-locked Python 3.12 CPU/CUDA inputs and offline checks for the **LLM-as-compiler execution bridge** and the retained NeuroVM reference primitives.

## Profiles

| Configuration | Use |
| --- | --- |
| [CPU](devcontainer.json) | Default; execution-bridge development, unit tests, and reference ALU/RAM checks without a GPU requirement |
| [CUDA / Docker](gpu/devcontainer.json) | NVIDIA host with Docker and NVIDIA Container Toolkit |
| [CUDA / Podman / WSL](gpu-podman/devcontainer.json) | WSL2 passthrough with Podman and NVIDIA CDI (`--device nvidia.com/gpu=all`, `/dev/dxg`) |

All profiles target Linux x86-64 and share the Ubuntu 24.04 base, `/opt/venv` Python 3.12 virtual environment, cache volume layout, and host CA certificate bootstrap. The CUDA profiles install PyTorch 2.10.0 CUDA 12.8 runtime wheels (`+cu128`).

## Compile-and-Execute Toolchain

These tools back the execution-bridge experiments in [docs/review/experiments](../docs/review/experiments) (see [PROJECT_DIRECTION.md](../docs/review/PROJECT_DIRECTION.md)):

| Component | Source | Purpose |
| --- | --- | --- |
| `clang` + `lld` (`wasm-ld`) | apt | C → wasm32 modules, and native C |
| `tcc` | apt | Millisecond C compile/run |
| `wabt` (`wat2wasm`, `wasm2wat`, `wasm-strip`) | apt | WebAssembly text/binary conversion |
| `bubblewrap` (`bwrap`) | apt | Namespace jail for model-generated programs |
| `wasmtime` | all locks | Wasm JIT runtime with fuel metering |
| `unicorn` | all locks | x86-64 emulation of raw machine code |
| `flash-linear-attention`, `fla-core` (+ `einops`) | CUDA lock only | Triton kernels for Qwen3.5's Gated DeltaNet layers. Without them, Transformers falls back to slow reference code. They import Triton, so they are kept out of the CPU locks. |

`bwrap` needs unprivileged user namespaces, so it depends on the container runtime:
- it works under rootless Podman;
- Docker's default seccomp profile may block it.

The smoke check reports this as `bubblewrap_namespaces` instead of failing.

## Start the Environment

Open the repository in VS Code and run **Dev Containers: Reopen in Container** for the default CPU configuration.

Or launch a specific profile via the Dev Containers CLI:

```sh
# Default CPU profile (Podman host)
devcontainer up --workspace-folder "$PWD" --docker-path podman \
  --buildkit never --gpu-availability none

# CUDA / Podman / WSL profile
devcontainer up --workspace-folder "$PWD" \
  --config "$PWD/.devcontainer/gpu-podman/devcontainer.json" \
  --docker-path podman --buildkit never --gpu-availability detect

# CUDA / Docker profile
devcontainer up --workspace-folder "$PWD" \
  --config "$PWD/.devcontainer/gpu/devcontainer.json" \
  --gpu-availability all
```

## Certificate Trust

`prepare-build.sh` exports public host root certificates from macOS keychains, Windows root stores (via WSL/PowerShell), or Linux system bundles into `.devcontainer/host-ca-certificates.crt` (git-ignored). The `Dockerfile` validates each certificate with OpenSSL and updates the Ubuntu trust store before running `apt` or `pip`.

To supply a custom PEM bundle explicitly on the host:

```sh
export NEURAL_COMPILER_CA_BUNDLE=/absolute/path/to/approved-ca-bundle.pem
```

## Dependency Locks

Dependencies are specified in [requirements.in](requirements.in), with CUDA-only additions in [gpu/requirements.in](gpu/requirements.in). They are locked with transitive SHA-256 hashes using `uv==0.8.22`.

| Lock | Target | Purpose |
| --- | --- | --- |
| [requirements.lock](requirements.lock) | Python 3.12, Linux x86-64, CPU | Default container and ordinary CI |
| [gpu/requirements.lock](gpu/requirements.lock) | Python 3.12, Linux x86-64, CUDA 12.8 | Docker/Podman CUDA containers |
| [requirements-py310.lock](requirements-py310.lock) | Python 3.10, Linux x86-64, CPU | Minimum-version compatibility CI; not a separate container profile |

To refresh the lockfiles inside a running container (existing pins are kept unless you add `--upgrade`):

```sh
uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-header --output-file .devcontainer/requirements.lock

uv pip compile .devcontainer/gpu/requirements.in --torch-backend cu128 \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-header --output-file .devcontainer/gpu/requirements.lock

uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.10 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-header --output-file .devcontainer/requirements-py310.lock
```

## Smoke Validation

[smoke.py](smoke.py) runs automatically as `postCreateCommand` (with `--toolchain`) without downloading pretrained model weights. It verifies:
- System CA trust store and TLS environment variables (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`).
- CPU or CUDA device availability (`NEURAL_COMPILER_DEVICE`).
- Deadzone bipolar snapping (`snap_bipolar`), exact Gated DeltaNet RAM row overwrite/read (`gdn_exact_write` / `gdn_exact_read`), synthetic `Qwen2ForCausalLM` forward wiring, and `Qwen3_5Config` availability.
- With `--toolchain`:
  - a C function compiled by clang to wasm32 and run in wasmtime with fuel;
  - a `wasm2wat`/`wat2wasm` round trip;
  - `tcc -run`;
  - a unicorn x86-64 emulation;
  - whether `bwrap` can create namespaces;
  - on CUDA, the fla kernel import.

Run unit tests, live network verification, and lint checks inside the container with:

```sh
python .devcontainer/smoke.py --tests --toolchain
python .devcontainer/smoke.py --network
shellcheck .devcontainer/prepare-build.sh
ruff check src tests .devcontainer/smoke.py
```
