FROM mcr.microsoft.com/devcontainers/base:1-ubuntu-24.04

ARG DEPENDENCY_LOCK=requirements.lock

USER root

ENV DEBIAN_FRONTEND=noninteractive

COPY host-ca-certificates.crt /tmp/host-ca-certificates.crt
RUN set -eu; \
    if grep -q -- '-----BEGIN CERTIFICATE-----' /tmp/host-ca-certificates.crt; then \
        directory=/usr/local/share/ca-certificates/host; \
        install -d "$directory"; \
        awk -v directory="$directory" \
            '{ sub(/\r$/, "") } /-----BEGIN CERTIFICATE-----/ { count++; path = sprintf("%s/host-%04d.crt", directory, count) } path { print > path } /-----END CERTIFICATE-----/ { close(path); path = "" }' \
            /tmp/host-ca-certificates.crt; \
        for certificate in "$directory"/*.crt; do \
            openssl x509 -noout -in "$certificate"; \
        done; \
        update-ca-certificates; \
    fi; \
    rm -f /tmp/host-ca-certificates.crt

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    UV_NATIVE_TLS=true

RUN sed -i 's|^URIs: http://|URIs: https://|' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        g++ \
        git \
        git-lfs \
        jq \
        make \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        shellcheck \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir pip==25.2 uv==0.8.22

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY ${DEPENDENCY_LOCK} /opt/neural-compiler/requirements.lock
RUN /opt/venv/bin/python -m pip install --no-cache-dir --require-hashes \
        --extra-index-url "${TORCH_INDEX_URL}" \
        -r /opt/neural-compiler/requirements.lock \
    && /opt/venv/bin/python -m pip check \
    && chown -R vscode:vscode /opt/venv \
    && install -d -o vscode -g vscode \
        /home/vscode/.cache/huggingface \
        /home/vscode/.cache/pip \
        /home/vscode/.cache/uv \
        /home/vscode/.cache/ccache

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/home/vscode/.cache/huggingface \
    XDG_CACHE_HOME=/home/vscode/.cache \
    CCACHE_DIR=/home/vscode/.cache/ccache \
    NEURAL_COMPILER_DEVICE=cpu \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1
