#!/bin/bash
set -euo pipefail

# =========================
# CONFIG
# =========================
RG="betterthannuvia"
ACR_NAME="betterthannuviaacr123456"
TAG="${TAG:-v1}"

PARSER_IMAGE_LOCAL="sales_automation:local"
PARSER_REPO="sales_automation-image"

# =========================
# PATHS ROBUSTI
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# =========================
# HELPERS
# =========================
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "❌ Comando non trovato: $1"; exit 1; }
}

build_local() {
  local dir="$1"
  local image="$2"

  if [[ ! -f "$dir/Dockerfile" ]]; then
    echo "❌ Dockerfile non trovato in: $dir"
    exit 1
  fi

  echo "🔨 Build (no-cache) $image da $dir"
  docker build --no-cache -t "$image" "$dir"
}

tag_and_push() {
  local image_local="$1"
  local repo="$2"
  local tag="$3"
  local remote="${ACR_LOGIN_SERVER}/${repo}:${tag}"

  echo "🏷️  Tag: $image_local -> $remote"
  docker tag "$image_local" "$remote"

  echo "📤 Push: $remote"
  docker push "$remote"
}

# =========================
# PRECHECK
# =========================
need_cmd az
need_cmd docker

echo "🔎 Verifico login Azure..."
az account show >/dev/null 2>&1 || {
  echo "❌ Non sei loggato. Esegui: az login --use-device-code"
  exit 1
}

echo "🔎 Verifico Resource Group: $RG"
az group show -n "$RG" >/dev/null 2>&1 || {
  echo "❌ Resource Group '$RG' non trovato."
  exit 1
}

# =========================
# ACR
# =========================
echo "🔐 Login su ACR..."
az acr login -n "$ACR_NAME"

ACR_LOGIN_SERVER="$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)"
echo "✅ ACR Login Server: $ACR_LOGIN_SERVER"

# =========================
# BUILD + PUSH
# =========================
build_local "$PARSER_DIR" "$PARSER_IMAGE_LOCAL"
tag_and_push "$PARSER_IMAGE_LOCAL" "$PARSER_REPO" "$TAG"

# =========================
# VERIFY
# =========================
echo "🏷️ Tag su repo '${PARSER_REPO}':"
az acr repository show-tags -n "$ACR_NAME" --repository "$PARSER_REPO" -o table

echo
echo "✅ DONE:"
echo " - ${ACR_LOGIN_SERVER}/${PARSER_REPO}:${TAG}"
