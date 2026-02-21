#!/usr/bin/env bash
# Scarica i log di conversazione da GET /admin/logs
#
# Uso:
#   bash test/get_logs.sh                          # ultimi 50 log
#   bash test/get_logs.sh --agent AG001            # filtra per agente
#   bash test/get_logs.sh --date 2026-02-20        # filtra per data
#   bash test/get_logs.sh --sender 393755116724    # filtra per telefono
#   bash test/get_logs.sh --limit 20 --offset 0
#   bash test/get_logs.sh --agent AG001 --date 2026-02-20 --limit 10
#   bash test/get_logs.sh --raw                    # JSON grezzo (no pretty print)

HOST="localhost"
PORT="9999"
AGENT=""
SENDER=""
DATE=""
LIMIT="50"
OFFSET="0"
RAW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)   HOST="$2";   shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        --agent)  AGENT="$2";  shift 2 ;;
        --sender) SENDER="$2"; shift 2 ;;
        --date)   DATE="$2";   shift 2 ;;
        --limit)  LIMIT="$2";  shift 2 ;;
        --offset) OFFSET="$2"; shift 2 ;;
        --raw)    RAW=true;    shift   ;;
        *) echo "Parametro sconosciuto: $1"; exit 1 ;;
    esac
done

# Costruisci query string
PARAMS="limit=${LIMIT}&offset=${OFFSET}"
[[ -n "$AGENT"  ]] && PARAMS="${PARAMS}&agent=${AGENT}"
[[ -n "$SENDER" ]] && PARAMS="${PARAMS}&sender=${SENDER}"
[[ -n "$DATE"   ]] && PARAMS="${PARAMS}&date=${DATE}"

URL="http://${HOST}:${PORT}/admin/logs?${PARAMS}"
echo "📥  GET ${URL}"
echo ""

if $RAW; then
    curl -s "$URL"
else
    curl -s "$URL" | python3 -c "
import sys, json

data = json.load(sys.stdin)
if 'error' in data:
    print('❌  Errore:', data['error'])
    sys.exit(1)

logs = data.get('logs', [])
total = data.get('total', 0)
offset = data.get('offset', 0)
print(f'📋  {total} log trovati (offset={offset})')
print('=' * 70)

for entry in logs:
    print(f\"[{entry['id']:>4}] {entry['ts']}  agent={entry['agent_code']}  sender={entry['sender_id']}  {entry['duration_ms']}ms\")
    print(f\"  👤 {entry['user_msg']}\")
    resp = (entry.get('response') or '').replace('\n', ' ')
    print(f\"  🤖 {resp[:120]}{'...' if len(resp or '') > 120 else ''}\")
    plan = entry.get('plan_json') or []
    if plan:
        tools = [t.get('tool','?') for t in plan if isinstance(t, dict)]
        print(f\"  🗂  Piano: {', '.join(tools)}\")
    print()
"
fi
