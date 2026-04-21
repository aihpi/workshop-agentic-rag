#!/usr/bin/env bash
#
# Offline SealedSecret generator — no kubectl / cluster access required.
#
# Usage:
#   scripts/seal-secret.sh <plaintext-secret.yaml> [sealed-out.yaml]
#
# Defaults:
#   output -> k8s/sealed-secret.yaml
#   cert   -> k8s/argocd/sealed-secrets.pub.pem (override via $SEALED_SECRETS_CERT)
#
# The cert is a PUBLIC key committed to the repo — safe to share. Only the
# sealed-secrets controller in-cluster can decrypt the resulting file.

set -euo pipefail

CERT="${SEALED_SECRETS_CERT:-k8s/argocd/sealed-secrets.pub.pem}"
INPUT="${1:?plaintext Secret manifest path required}"
OUTPUT="${2:-k8s/sealed-secret.yaml}"

if [[ ! -f "$CERT" ]]; then
  echo "ERROR: cert not found at $CERT" >&2
  echo "  Get it once from someone with cluster access:" >&2
  echo "  kubeseal --fetch-cert --controller-namespace sealed-secrets --controller-name sealed-secrets-controller > $CERT" >&2
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input file not found: $INPUT" >&2
  exit 1
fi

kubeseal --cert "$CERT" --format yaml < "$INPUT" > "$OUTPUT"
echo "Sealed: $INPUT -> $OUTPUT"
