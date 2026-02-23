#!/bin/bash
set -euo pipefail

# =========================
# Build immagine + deploy su Azure (lancia step 1 e 2 in cascata)
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  STEP 1: Upload immagine su Azure ACR"
echo "=========================================="
bash "$SCRIPT_DIR/1 - upload_image_to_azure.sh"

echo
echo "=========================================="
echo "  STEP 2: Deploy container su Azure"
echo "=========================================="
bash "$SCRIPT_DIR/2 - deploy_container_on_azure.sh"
