#!/usr/bin/env bash
# Chiama POST /admin/reset: ricrea DB ordini e svuota i checkpoint LangGraph.

HOST="${1:-localhost}"
PORT="${2:-9999}"
URL="http://${HOST}:${PORT}/admin/reset"

echo "🔄  Chiamata a ${URL} ..."
curl -s -X POST "${URL}" | python3 -m json.tool
