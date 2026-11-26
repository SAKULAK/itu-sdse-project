#!/usr/bin/env bash
# Deploy (or update) the model-serving image on Azure Container Apps.
#
# The image must already be pushed to ghcr.io/<owner>/model-serve:<tag> and be
# public (or see the registries note in infra/azure/app.yaml).
#
# Usage:  ./scripts/azure/deploy.sh <github-owner> <image-tag> [--dry-run]
#   --dry-run  only render infra/azure/app.yaml and print it; touches nothing.
#
# Overrides (env vars or .env.azure): AZ_RG, AZ_LOC, AZ_ENV, AZ_APP
# Credentials: same as login.sh (env vars, else .env.azure).
set -euo pipefail

usage() { echo "Usage: $0 <github-owner> <image-tag> [--dry-run]" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
OWNER="$(tr '[:upper:]' '[:lower:]' <<<"$1")"   # GHCR paths must be lowercase
TAG="$2"
DRY_RUN=false
[[ "${3:-}" == "--dry-run" ]] && DRY_RUN=true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${AZ_ENV_FILE:-$ROOT/.env.azure}"
TEMPLATE="$ROOT/infra/azure/app.yaml"

# Pick up AZ_RG / AZ_LOC / subscription from .env.azure unless already set.
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key val; do
    [[ "$key" =~ ^[A-Z_]+$ ]] || continue
    [[ -n "${!key:-}" ]] && continue
    val="${val%\'}"; val="${val#\'}"
    export "$key=$val"
  done <"$ENV_FILE"
fi

RG="${AZ_RG:-mlops-itu-rg}"
LOC="${AZ_LOC:-norwayeast}"
ENV_NAME="${AZ_ENV:-mlops-itu-env}"
APP="${AZ_APP:-mlops-itu-jtk}"
SUB="${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID not set (run bootstrap-sp.sh or export it)}"

az_tsv() { az "$@" -o tsv | tr -d '\r'; }   # az on Windows emits CRLF

# Render the template into a temp file (the template itself stays untouched).
RENDERED="$(mktemp "${TMPDIR:-/tmp}/app.rendered.XXXXXX.yaml")"
trap 'rm -f "$RENDERED"' EXIT
sed \
  -e "s|__OWNER__|$OWNER|g" \
  -e "s|__TAG__|$TAG|g" \
  -e "s|__SUBSCRIPTION_ID__|$SUB|g" \
  -e "s|__RESOURCE_GROUP__|$RG|g" \
  -e "s|__ENVIRONMENT__|$ENV_NAME|g" \
  "$TEMPLATE" >"$RENDERED"

if grep -q '__[A-Z_]*__' "$RENDERED"; then
  echo "Unfilled placeholders left in rendered spec:" >&2
  grep -n '__[A-Z_]*__' "$RENDERED" >&2
  exit 1
fi

if $DRY_RUN; then
  cat "$RENDERED"
  exit 0
fi

# az.exe on Windows needs a Windows path, not a Git Bash /tmp path.
RENDERED_ARG="$RENDERED"
command -v cygpath >/dev/null 2>&1 && RENDERED_ARG="$(cygpath -w "$RENDERED")"

"$HERE/login.sh"

az extension show --name containerapp >/dev/null 2>&1 \
  || az extension add --name containerapp --upgrade --only-show-errors

if ! az containerapp env show --name "$ENV_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  echo ">>> Creating Container Apps environment $ENV_NAME (takes a few minutes, first time only)..."
  az containerapp env create --name "$ENV_NAME" --resource-group "$RG" --location "$LOC" --output none
fi

if az containerapp show --name "$APP" --resource-group "$RG" >/dev/null 2>&1; then
  echo ">>> Updating $APP to ghcr.io/$OWNER/model-serve:$TAG..."
  az containerapp update --name "$APP" --resource-group "$RG" --yaml "$RENDERED_ARG" --output none
else
  echo ">>> Creating $APP with ghcr.io/$OWNER/model-serve:$TAG..."
  az containerapp create --name "$APP" --resource-group "$RG" --yaml "$RENDERED_ARG" --output none
fi

FQDN="$(az_tsv containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn)"
echo ">>> Deployed: https://$FQDN  (health: https://$FQDN/health)"
