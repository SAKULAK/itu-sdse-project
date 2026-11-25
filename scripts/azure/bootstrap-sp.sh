#!/usr/bin/env bash
# One-time setup: creates a service principal that scripts/CI can use to log in
# to Azure without a browser, and writes its credentials to .env.azure.
#
# Requires an interactive login first (`az login`). Re-running it rotates the secret.
#
# Usage:  ./scripts/azure/bootstrap-sp.sh
# Overrides (env vars): AZ_RG, AZ_LOC, AZ_SP_NAME, AZ_ENV_FILE, AZ_SP_YEARS
set -euo pipefail

# Git Bash on Windows rewrites args starting with "/" into Windows paths,
# which breaks `--scopes /subscriptions/...`. Disable that.
export MSYS_NO_PATHCONV=1

RG="${AZ_RG:-mlops-itu-rg}"
LOC="${AZ_LOC:-norwayeast}"
SP_NAME="${AZ_SP_NAME:-mlops-itu-deployer}"
SP_YEARS="${AZ_SP_YEARS:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${AZ_ENV_FILE:-$ROOT/.env.azure}"

# az on Windows emits CRLF in tsv output; strip the \r.
az_tsv() { az "$@" -o tsv | tr -d '\r'; }

if ! az account show >/dev/null 2>&1; then
  echo ">>> Not logged in - opening browser login..."
  az login >/dev/null
fi

SUB_ID="$(az_tsv account show --query id)"
SUB_NAME="$(az_tsv account show --query name)"
TENANT_ID="$(az_tsv account show --query tenantId)"
echo ">>> Using subscription: $SUB_NAME ($SUB_ID)"
echo "    (switch with: az account set --subscription <id>)"

echo ">>> Registering resource providers (needs subscription-level rights, done once)..."
az provider register --namespace Microsoft.App >/dev/null
az provider register --namespace Microsoft.OperationalInsights >/dev/null

echo ">>> Ensuring resource group $RG in $LOC..."
az group create --name "$RG" --location "$LOC" >/dev/null

SCOPE="/subscriptions/$SUB_ID/resourceGroups/$RG"
echo ">>> Creating service principal '$SP_NAME' with Contributor on $SCOPE..."
if ! SP_OUT="$(az_tsv ad sp create-for-rbac \
      --name "$SP_NAME" \
      --role Contributor \
      --scopes "$SCOPE" \
      --years "$SP_YEARS" \
      --query "[appId, password]")"; then
  cat >&2 <<'EOF'

!!! Could not create the service principal.

EOF
  exit 1
fi

CLIENT_ID="$(sed -n 1p <<<"$SP_OUT")"
CLIENT_SECRET="$(sed -n 2p <<<"$SP_OUT")"

umask 077
cat >"$ENV_FILE" <<EOF
# Azure service principal credentials - DO NOT COMMIT.
# Created by scripts/azure/bootstrap-sp.sh on $(date -u +%Y-%m-%d). Secret expires after $SP_YEARS year(s).
AZURE_CLIENT_ID='$CLIENT_ID'
AZURE_CLIENT_SECRET='$CLIENT_SECRET'
AZURE_TENANT_ID='$TENANT_ID'
AZURE_SUBSCRIPTION_ID='$SUB_ID'
AZ_RG='$RG'
AZ_LOC='$LOC'
EOF

echo ">>> Wrote credentials to $ENV_FILE"
echo ">>> Log in with:  ./scripts/azure/login.sh"
