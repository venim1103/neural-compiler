#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DEST="${SCRIPT_DIR}/host-ca-certificates.crt"
CERT_TEMP="$(mktemp "${CERT_DEST}.XXXXXX")"
trap 'rm -f "$CERT_TEMP"' EXIT

has_certificate() {
    [[ -s "$1" ]] && grep -q -- '-----BEGIN CERTIFICATE-----' "$1"
}

export_macos() {
    local keychain
    while IFS= read -r keychain; do
        keychain="${keychain#*\"}"
        keychain="${keychain%\"*}"
        if [[ -r "$keychain" ]]; then
            security find-certificate -a -p "$keychain" >> "$CERT_TEMP"
        fi
    done < <(security list-keychains -d user; security list-keychains -d system)
}

export_windows() {
    local destination powershell_command
    if grep -qi microsoft /proc/version 2>/dev/null; then
        destination="$(wslpath -w "$CERT_TEMP")"
    else
        destination="$(cygpath -w "$CERT_TEMP")"
    fi
    read -r -d '' powershell_command <<'POWERSHELL' || true
$ErrorActionPreference = "Stop"
$destination = [Console]::In.ReadLine()
$certificates = Get-ChildItem Cert:\CurrentUser\Root, Cert:\LocalMachine\Root
$pem = foreach ($certificate in $certificates) {
    $base64 = [Convert]::ToBase64String(
        $certificate.RawData,
        [Base64FormattingOptions]::InsertLineBreaks
    )
    "-----BEGIN CERTIFICATE-----`n$base64`n-----END CERTIFICATE-----`n"
}
[IO.File]::WriteAllText($destination, ($pem -join ""), [Text.Encoding]::ASCII)
POWERSHELL
    printf '%s\n' "$destination" | \
        powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$powershell_command"
}

export_host_bundle() {
    local bundle
    for bundle in \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/ssl/certs/ca-bundle.crt \
        /etc/pki/tls/certs/ca-bundle.crt \
        /mingw64/etc/ssl/certs/ca-bundle.crt; do
        if [[ -r "$bundle" ]]; then
            cp "$bundle" "$CERT_TEMP"
            return 0
        fi
    done
    return 1
}

if [[ -n "${NEURAL_COMPILER_CA_BUNDLE:-}" ]]; then
    if ! has_certificate "$NEURAL_COMPILER_CA_BUNDLE"; then
        printf 'Error: NEURAL_COMPILER_CA_BUNDLE must name a readable PEM certificate bundle.\n' >&2
        exit 1
    fi
    cp "$NEURAL_COMPILER_CA_BUNDLE" "$CERT_TEMP"
elif command -v security >/dev/null 2>&1; then
    export_macos || printf 'Warning: macOS certificate export failed; trying the system bundle.\n' >&2
elif command -v powershell.exe >/dev/null 2>&1; then
    export_windows || printf 'Warning: Windows certificate export failed; trying the system bundle.\n' >&2
fi

if ! has_certificate "$CERT_TEMP"; then
    export_host_bundle || true
fi
if ! has_certificate "$CERT_TEMP"; then
    printf 'Warning: no host certificates found; the image keeps its default trust store.\n' >&2
    : > "$CERT_TEMP"
fi
if grep -q -- 'PRIVATE KEY' "$CERT_TEMP"; then
    printf 'Error: the CA bundle must not contain private keys.\n' >&2
    exit 1
fi

chmod 0644 "$CERT_TEMP"
mv "$CERT_TEMP" "$CERT_DEST"
count="$(grep -c -- '-----BEGIN CERTIFICATE-----' "$CERT_DEST" || true)"
printf 'Prepared %s host certificates for the container build.\n' "$count"
