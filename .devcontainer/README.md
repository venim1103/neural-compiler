# Development Container (`neural-compiler`)

This setup adapts the multi-profile Dev Container and certificate-trust architecture from `llm-loop-fix`, `micro_classifier`, and `embedded-jev`. It provides hash-locked Python 3.12 CPU/CUDA inputs and offline numerical/wiring checks for the **NeuroVM** weight-compiled looped neural virtual machine and compiler stack.

## Profiles

| Configuration | Use |
| --- | --- |
| [CPU](devcontainer.json) | Default; weight-compiled ALU/RAM circuit synthesis, unit tests, and looped state verification without a GPU requirement |
| [CUDA / Docker](gpu/devcontainer.json) | NVIDIA host with Docker and NVIDIA Container Toolkit |
| [CUDA / Podman / WSL](gpu-podman/devcontainer.json) | WSL2 passthrough with Podman and NVIDIA CDI (`--device nvidia.com/gpu=all`, `/dev/dxg`) |

All profiles target Linux x86-64 and share the Ubuntu 24.04 base, `/opt/venv` Python 3.12 virtual environment, cache volume layout, and host CA certificate bootstrap. The CUDA profiles install PyTorch 2.10.0 CUDA 12.8 runtime wheels (`+cu128`).

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

Dependencies are specified in [requirements.in](requirements.in) and locked with transitive SHA-256 hashes using `uv==0.8.22`.

| Lock | Target | Purpose |
| --- | --- | --- |
| [requirements.lock](requirements.lock) | Python 3.12, Linux x86-64, CPU | Default container and ordinary CI |
| [gpu/requirements.lock](gpu/requirements.lock) | Python 3.12, Linux x86-64, CUDA 12.8 | Docker/Podman CUDA containers |
| [requirements-py310.lock](requirements-py310.lock) | Python 3.10, Linux x86-64, CPU | Minimum-version compatibility CI; not a separate container profile |

To refresh the lockfiles inside a running container:

```sh
uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/requirements.lock

uv pip compile .devcontainer/requirements.in --torch-backend cu128 \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/gpu/requirements.lock

uv pip compile .devcontainer/requirements.in --torch-backend cpu \
  --python-version 3.10 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file .devcontainer/requirements-py310.lock
```

## Smoke Validation

[smoke.py](smoke.py) runs automatically as `postCreateCommand` without downloading pretrained model weights. It verifies:
- System CA trust store and TLS environment variables (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`).
- CPU or CUDA device availability (`NEURAL_COMPILER_DEVICE`).
- Deadzone bipolar snapping (`snap_bipolar`), exact Gated DeltaNet RAM row overwrite/read (`gdn_exact_write` / `gdn_exact_read`), synthetic `Qwen2ForCausalLM` forward wiring, and `Qwen3_5Config` availability.

Run unit tests, live network verification, and lint checks inside the container with:

```sh
python .devcontainer/smoke.py --tests
python .devcontainer/smoke.py --network
shellcheck .devcontainer/prepare-build.sh
ruff check src tests .devcontainer/smoke.py
```
