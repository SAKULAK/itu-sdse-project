#!/usr/bin/env bash
# Non-interactive Azure login using a service principal.
#
# Reads AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID / AZURE_SUBSCRIPTION_ID
# from the environment (e.g. CI secrets). 
# Usage:  ./scripts/azure/login.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${AZ_ENV_FILE:-$ROOT/.env.azure}"

if [[ -z "${AZURE_CLIENT_ID:-}" || -z "${AZURE_CLIENT_SECRET:-}" || -z "${AZURE_TENANT_ID:-}" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    . "$ENV_FILE"
    set +a
  fi
fi

missing=()
for v in AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID; do
  [[ -n "${!v:-}" ]] || missing+=("$v")
done
if (( ${#missing[@]} )); then
  echo "Missing: ${missing[*]}" >&2
  echo "Set them in the environment or run ./scripts/azure/bootstrap-sp.sh to create $ENV_FILE" >&2
  exit 1
fi

# A freshly created service principal's role assignment can take a few minutes
# to propagate; until then login fails with "No subscriptions found". Retry.
attempts="${AZ_LOGIN_ATTEMPTS:-10}"
for ((i = 1; i <= attempts; i++)); do
  if az login --service-principal \
      --username "$AZURE_CLIENT_ID" \
      --password "$AZURE_CLIENT_SECRET" \
      --tenant "$AZURE_TENANT_ID" \
      --output none 2>/tmp/az-login-err.$$; then
    rm -f /tmp/az-login-err.$$
    break
  fi
  if (( i == attempts )) || ! grep -q "No subscriptions found" /tmp/az-login-err.$$; then
    cat /tmp/az-login-err.$$ >&2; rm -f /tmp/az-login-err.$$
    exit 1
  fi
  echo "Role assignment not visible yet (attempt $i/$attempts), retrying in 30s..." >&2
  sleep 30
done

az account set --subscription "$AZURE_SUBSCRIPTION_ID"
echo "Logged in as service principal $AZURE_CLIENT_ID on subscription $(az account show --query name -o tsv | tr -d '\r')"
