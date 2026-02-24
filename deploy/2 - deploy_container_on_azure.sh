#!/bin/bash
set -euo pipefail

# ============================================================
# DEPLOY sales_automation su Azure Container Apps
# - HTTP ingress esterno su porta 9999 (webhook WhatsApp)
# - Crea app con dummy image se manca
# - Assegna system identity
# - Assegna AcrPull su ACR e attende propagazione
# - Configura registry con identity system
# - Aggiorna immagine + secrets + env vars (secretref)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PARSER_DIR/.env"

RG="sales-automation"          # Resource Group del Container App
ACR_RG="betterthannuvia"       # Resource Group dove sta l'ACR
LOCATION="italynorth"
ENV_NAME="btn-env"

APP_NAME="sales-automation-container"

WEAVIATE_APP_NAME="weaviate"
WEAVIATE_IMAGE="cr.weaviate.io/semitechnologies/weaviate:1.28.4"

ACR_NAME="betterthannuviaacr123456"
IMAGE_REPO="sales_automation-image"
IMAGE_TAG="${IMAGE_TAG:-v1}"

DUMMY_IMAGE="mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"

MIN_REPLICAS="${MIN_REPLICAS:-1}"
MAX_REPLICAS="${MAX_REPLICAS:-1}"

# ----------------------------
# Helpers
# ----------------------------
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing command: $1"; exit 1; }; }
die() { echo "❌ $*"; exit 1; }

require_var() {
  local n="$1"
  [[ -n "${!n:-}" ]] || die "Missing env var in .env: $n"
}

# dotenv-safe loader (NON usare source)
get_env() {
  local key="$1"
  local line val
  line="$(grep -m1 -E "^${key}=" "$ENV_FILE" || true)"
  [[ -n "$line" ]] || return 1
  val="${line#*=}"
  val="${val//$'\r'/}"
  val="${val%\"}"; val="${val#\"}"
  val="${val%\'}"; val="${val#\'}"
  printf '%s' "$val"
}

get_env_default() {
  local key="$1"
  local def="$2"
  local v
  v="$(get_env "$key" 2>/dev/null || true)"
  if [[ -z "${v:-}" ]]; then
    printf '%s' "$def"
  else
    printf '%s' "$v"
  fi
}

wait_provisioning() {
  local app="$1"
  local timeout_sec="${2:-600}"
  local interval_sec="${3:-5}"

  echo "⏳ Attendo provisioning di: $app (timeout ${timeout_sec}s)"
  local start now state
  start="$(date +%s)"

  while true; do
    state="$(az containerapp show -g "$RG" -n "$app" --query properties.provisioningState -o tsv 2>/dev/null || echo "")"
    [[ -n "$state" ]] && echo "  provisioningState=$state"

    if [[ "$state" == "Succeeded" ]]; then return 0; fi
    if [[ "$state" == "Failed" ]]; then return 1; fi

    now="$(date +%s)"
    if (( now - start >= timeout_sec )); then
      echo "⚠️ Timeout provisioning (state=$state)"
      return 2
    fi
    sleep "$interval_sec"
  done
}

dump_diagnostics() {
  local app="$1"
  echo
  echo "================ DIAGNOSTICA ================"
  echo "App: $app"
  echo "- provisioningState:"
  az containerapp show -g "$RG" -n "$app" --query properties.provisioningState -o tsv 2>/dev/null || true
  echo
  echo "- latestRevisionName:"
  az containerapp show -g "$RG" -n "$app" --query properties.latestRevisionName -o tsv 2>/dev/null || true
  echo
  echo "- revision list:"
  az containerapp revision list -g "$RG" -n "$app" -o table 2>/dev/null || true
  echo
  echo "- ultimi log (se disponibili):"
  az containerapp logs show -g "$RG" -n "$app" --tail 200 2>/dev/null || true
  echo "============================================="
  echo
}

wait_acrpull() {
  local principal_id="$1"
  local acr_id="$2"
  local timeout_sec="${3:-300}"
  local interval_sec="${4:-5}"

  echo "⏳ Attendo propagazione ruolo AcrPull su ACR (timeout ${timeout_sec}s)"
  local start now count
  start="$(date +%s)"

  while true; do
    count="$(az role assignment list --assignee "$principal_id" --scope "$acr_id" \
      --query "[?roleDefinitionName=='AcrPull'] | length(@)" -o tsv 2>/dev/null || echo "0")"
    echo "  AcrPull count=$count"
    if [[ "$count" == "1" ]]; then
      echo "✅ AcrPull presente"
      return 0
    fi

    now="$(date +%s)"
    if (( now - start >= timeout_sec )); then
      echo "⚠️ Timeout: AcrPull non risulta ancora presente (può propagarsi più tardi)"
      return 1
    fi
    sleep "$interval_sec"
  done
}

# ----------------------------
# Precheck
# ----------------------------
need_cmd az
az extension add --name containerapp --upgrade >/dev/null 2>&1 || true

az account show >/dev/null 2>&1 || die "Run: az login --use-device-code"
az group show -n "$RG" >/dev/null 2>&1 || die "RG not found: $RG"

[[ -f "$ENV_FILE" ]] || die ".env not found: $ENV_FILE"
echo "📥 Carico variabili da: $ENV_FILE"

# ----------------------------
# Load .env (SAFE)
# ----------------------------
OPENAI_API_KEY="$(get_env OPENAI_API_KEY || true)"
GEMINI_API_KEY="$(get_env GEMINI_API_KEY || true)"
AZURE_SEARCH_ENDPOINT="$(get_env AZURE_SEARCH_ENDPOINT || true)"
AZURE_SEARCH_ADMIN_KEY="$(get_env AZURE_SEARCH_ADMIN_KEY || true)"
GMAIL_FROM="$(get_env GMAIL_FROM || true)"
GMAIL_APP_PASSWORD="$(get_env GMAIL_APP_PASSWORD || true)"

PLANNER_PROVIDER="$(get_env_default PLANNER_PROVIDER openai)"
GENERIC_PROVIDER="$(get_env_default GENERIC_PROVIDER openai)"
OPENAI_MODEL_PLANNER="$(get_env_default OPENAI_MODEL_PLANNER gpt-4o-mini)"
OPENAI_MODEL_GENERIC="$(get_env_default OPENAI_MODEL_GENERIC gpt-4o)"
GEMINI_MODEL_PLANNER="$(get_env_default GEMINI_MODEL_PLANNER "")"
GEMINI_MODEL_GENERIC="$(get_env_default GEMINI_MODEL_GENERIC "")"

DEPLOY_TS="$(date -u +%Y%m%d%H%M%S)"

# Required
require_var "OPENAI_API_KEY"
require_var "AZURE_SEARCH_ENDPOINT"
require_var "AZURE_SEARCH_ADMIN_KEY"
require_var "GMAIL_FROM"
require_var "GMAIL_APP_PASSWORD"

# ----------------------------
# Providers + env
# ----------------------------
az provider register --namespace Microsoft.App >/dev/null
az provider register --namespace Microsoft.OperationalInsights >/dev/null

if ! az containerapp env show -g "$RG" -n "$ENV_NAME" >/dev/null 2>&1; then
  echo "🆕 Creo Container Apps Environment: $ENV_NAME"
  az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOCATION" >/dev/null
fi

# ----------------------------
# Deploy Weaviate (immagine pubblica)
# - Ingress INTERNO (l'app popola Weaviate allo startup automaticamente)
# - gRPC porta 50051 via additionalPortMappings
# ----------------------------
WEAVIATE_ENV_VARS=(
  "QUERY_DEFAULTS_LIMIT=25"
  "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true"
  "PERSISTENCE_DATA_PATH=/var/lib/weaviate"
  "DEFAULT_VECTORIZER_MODULE=text2vec-openai"
  "ENABLE_MODULES=text2vec-openai"
  "CLUSTER_HOSTNAME=node1"
  "OPENAI_APIKEY=$OPENAI_API_KEY"
)

echo "🔷 Deploy Weaviate..."
if ! az containerapp show -g "$RG" -n "$WEAVIATE_APP_NAME" >/dev/null 2>&1; then
  echo "🆕 Creo container Weaviate (ingress internal, porta 8080)..."
  az containerapp create \
    -g "$RG" \
    -n "$WEAVIATE_APP_NAME" \
    --environment "$ENV_NAME" \
    --image "$WEAVIATE_IMAGE" \
    --ingress internal \
    --target-port 8080 \
    --min-replicas 1 \
    --max-replicas 1 \
    --cpu 1.0 --memory 2.0Gi \
    --env-vars "${WEAVIATE_ENV_VARS[@]}" \
    --no-wait >/dev/null

  if ! wait_provisioning "$WEAVIATE_APP_NAME" 900 5; then
    dump_diagnostics "$WEAVIATE_APP_NAME"
    die "Provisioning Weaviate fallito"
  fi
  echo "✅ Weaviate creato (ingress interno)"
else
  echo "♻️ Weaviate esiste, aggiorno immagine + env vars..."
  az containerapp update -g "$RG" -n "$WEAVIATE_APP_NAME" \
    --image "$WEAVIATE_IMAGE" \
    --set-env-vars "${WEAVIATE_ENV_VARS[@]}" \
    --no-wait >/dev/null

  if ! wait_provisioning "$WEAVIATE_APP_NAME" 900 5; then
    dump_diagnostics "$WEAVIATE_APP_NAME"
    die "Provisioning Weaviate fallito durante update"
  fi
  echo "✅ Weaviate aggiornato"
fi

# Configura ingress interno + porta gRPC 50051
echo "🔒 Configuro Weaviate: ingress interno + porta gRPC 50051..."
WEAVIATE_ID="$(az containerapp show -g "$RG" -n "$WEAVIATE_APP_NAME" --query id -o tsv)"
echo "  Weaviate ID: $WEAVIATE_ID"

if ! az rest --method patch --url "${WEAVIATE_ID}?api-version=2024-03-01" --body '{
  "properties": {
    "configuration": {
      "ingress": {
        "external": false,
        "targetPort": 8080,
        "transport": "Auto",
        "additionalPortMappings": [
          {
            "external": false,
            "targetPort": 50051,
            "exposedPort": 50051
          }
        ]
      }
    }
  }
}'; then
  echo "⚠️ az rest patch fallito — provo con ingress CLI (senza gRPC)..."
  az containerapp ingress enable -g "$RG" -n "$WEAVIATE_APP_NAME" \
    --type internal --target-port 8080 --transport auto || true
fi

if ! wait_provisioning "$WEAVIATE_APP_NAME" 300 5; then
  echo "⚠️ Weaviate provisioning post-patch non completato (può essere ok)"
fi

# Verifica configurazione ingress finale
echo "🔍 Verifica ingress Weaviate:"
az containerapp show -g "$RG" -n "$WEAVIATE_APP_NAME" \
  --query '{external: properties.configuration.ingress.external, targetPort: properties.configuration.ingress.targetPort, additionalPorts: properties.configuration.ingress.additionalPortMappings}' \
  -o json 2>/dev/null || true

# ----------------------------
# ACR info + sanity checks
# ----------------------------
ACR_SERVER="$(az acr show -n "$ACR_NAME" --resource-group "$ACR_RG" --query loginServer -o tsv)"
ACR_ID="$(az acr show -n "$ACR_NAME" --resource-group "$ACR_RG" --query id -o tsv)"
IMAGE="${ACR_SERVER}/${IMAGE_REPO}:${IMAGE_TAG}"
echo "🖼️ Immagine: $IMAGE"

echo "🔎 Verifico che il tag esista su ACR..."
if ! az acr repository show-tags -n "$ACR_NAME" --repository "$IMAGE_REPO" \
  --query "[?@=='$IMAGE_TAG'] | length(@)" -o tsv | grep -q '^1$'; then
  echo "⚠️ Tag '$IMAGE_TAG' non trovato in '$IMAGE_REPO'. Tag disponibili:"
  az acr repository show-tags -n "$ACR_NAME" --repository "$IMAGE_REPO" -o table || true
  die "Immagine/tag non presente su ACR: $IMAGE"
fi
echo "✅ Tag trovato"

# ----------------------------
# Create app if missing (HTTP ingress esterno su porta 9999)
# ----------------------------
if ! az containerapp show -g "$RG" -n "$APP_NAME" >/dev/null 2>&1; then
  echo "🆕 Creo app con dummy image (HTTP ingress esterno su 9999)..."
  az containerapp create \
    -g "$RG" \
    -n "$APP_NAME" \
    --environment "$ENV_NAME" \
    --image "$DUMMY_IMAGE" \
    --ingress external \
    --target-port 9999 \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --no-wait >/dev/null

  if ! wait_provisioning "$APP_NAME" 900 5; then
    dump_diagnostics "$APP_NAME"
    die "Provisioning fallito durante create"
  fi
fi

# ----------------------------
# Identity + AcrPull + Registry
# ----------------------------
echo "👤 Assegno system identity..."
az containerapp identity assign -g "$RG" -n "$APP_NAME" --system-assigned >/dev/null

PRINCIPAL_ID="$(az containerapp show -g "$RG" -n "$APP_NAME" --query identity.principalId -o tsv)"
[[ -n "$PRINCIPAL_ID" ]] || die "principalId vuoto (identity non assegnata?)"
echo "✅ principalId: $PRINCIPAL_ID"

echo "🔐 Assegno ruolo AcrPull su ACR (idempotente)..."
set +e
az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role AcrPull \
  --scope "$ACR_ID" >/dev/null 2>&1
set -e

wait_acrpull "$PRINCIPAL_ID" "$ACR_ID" 600 5 || true

echo "📦 Imposto registry sull'app usando identity system..."
az containerapp registry set -g "$RG" -n "$APP_NAME" --server "$ACR_SERVER" --identity system >/dev/null

# ----------------------------
# Update image (forza nuova revisione)
# ----------------------------
echo "♻️ Aggiorno immagine..."
az containerapp update -g "$RG" -n "$APP_NAME" --image "$IMAGE" --no-wait >/dev/null

if ! wait_provisioning "$APP_NAME" 900 5; then
  dump_diagnostics "$APP_NAME"
  die "Provisioning fallito durante update image (pull ACR?)"
fi
echo "✅ Immagine aggiornata"

# ----------------------------
# Secrets (PRIMA degli env var secretref)
# ----------------------------
echo "🔑 Imposto secrets..."
az containerapp secret set -g "$RG" -n "$APP_NAME" \
  --secrets "openai-api-key=$OPENAI_API_KEY" >/dev/null
az containerapp secret set -g "$RG" -n "$APP_NAME" \
  --secrets "search-admin-key=$AZURE_SEARCH_ADMIN_KEY" >/dev/null
az containerapp secret set -g "$RG" -n "$APP_NAME" \
  --secrets "gmail-app-password=$GMAIL_APP_PASSWORD" >/dev/null

# GEMINI opzionale: aggiunto solo se presente nel .env
if [[ -n "$GEMINI_API_KEY" ]]; then
  az containerapp secret set -g "$RG" -n "$APP_NAME" \
    --secrets "gemini-api-key=$GEMINI_API_KEY" >/dev/null
fi
echo "✅ Secrets impostati"

# ----------------------------
# Env vars (secretref + normali) + force new revision
# ----------------------------
echo "🌱 Aggiorno env vars e forzo nuova revisione..."

UPDATE_ENV_VARS=(
  "OPENAI_API_KEY=secretref:openai-api-key"
  "AZURE_SEARCH_ADMIN_KEY=secretref:search-admin-key"
  "GMAIL_APP_PASSWORD=secretref:gmail-app-password"

  "AZURE_SEARCH_ENDPOINT=$AZURE_SEARCH_ENDPOINT"
  "GMAIL_FROM=$GMAIL_FROM"

  "PLANNER_PROVIDER=$PLANNER_PROVIDER"
  "GENERIC_PROVIDER=$GENERIC_PROVIDER"
  "OPENAI_MODEL_PLANNER=$OPENAI_MODEL_PLANNER"
  "OPENAI_MODEL_GENERIC=$OPENAI_MODEL_GENERIC"
  "GEMINI_MODEL_PLANNER=$GEMINI_MODEL_PLANNER"
  "GEMINI_MODEL_GENERIC=$GEMINI_MODEL_GENERIC"

  "WEAVIATE_URL=http://weaviate:80"
  "WEAVIATE_GRPC_PORT=50051"
  "WEAVIATE_SKIP_INIT=true"

  "DEPLOY_TS=$DEPLOY_TS"
)
[[ -n "$GEMINI_API_KEY" ]] && UPDATE_ENV_VARS+=("GEMINI_API_KEY=secretref:gemini-api-key")

az containerapp update -g "$RG" -n "$APP_NAME" \
  --set-env-vars "${UPDATE_ENV_VARS[@]}" \
  --no-wait >/dev/null

if ! wait_provisioning "$APP_NAME" 900 5; then
  dump_diagnostics "$APP_NAME"
  die "Provisioning fallito durante update env vars"
fi

CUR_IMAGE="$(az containerapp show -g "$RG" -n "$APP_NAME" --query properties.template.containers[0].image -o tsv)"
CUR_REV="$(az containerapp show -g "$RG" -n "$APP_NAME" --query properties.latestRevisionName -o tsv)"
FQDN="$(az containerapp show -g "$RG" -n "$APP_NAME" --query properties.configuration.ingress.fqdn -o tsv)"

echo
# NB: Weaviate viene popolato automaticamente dall'app allo startup
#     (vedi _init_weaviate in endpoint.py)

echo "✅ DEPLOY COMPLETATO"
echo "APP:       $APP_NAME"
echo "IMG:       $CUR_IMAGE"
echo "REV:       $CUR_REV"
echo "URL:       https://$FQDN"
echo "WEAVIATE:  $WEAVIATE_APP_NAME (internal, HTTP :80 + gRPC :50051)"
echo
echo "Webhook WhatsApp da configurare su Meta:"
echo "  https://$FQDN/whatsapp"
echo
echo "Logs:"
echo "  az containerapp logs show -g $RG -n $APP_NAME --follow"
echo "  az containerapp logs show -g $RG -n $WEAVIATE_APP_NAME --follow"
