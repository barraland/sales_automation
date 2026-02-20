#!/usr/bin/env python3
"""
Scarica e stampa i log di conversazione dal server.

Uso:
    python test/fetch_logs.py                        # AG001, ultimi 20
    python test/fetch_logs.py --agent AG002
    python test/fetch_logs.py --limit 50
    python test/fetch_logs.py --date 2026-02-20
    python test/fetch_logs.py --out logs.json        # salva su file
    python test/fetch_logs.py --clear                # cancella log AG001
"""

import argparse
import json
import sys
import requests
from datetime import date

DEFAULT_AGENT = "AG001"
DEFAULT_LIMIT = 20


def fetch(base_url: str, agent: str, limit: int, date_filter: str) -> list:
    params = {"agent": agent, "limit": limit}
    if date_filter:
        params["date"] = date_filter
    r = requests.get(f"{base_url}/admin/logs", params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("logs", [])


def clear(base_url: str, agent: str):
    r = requests.delete(f"{base_url}/admin/logs", params={"agent": agent}, timeout=10)
    r.raise_for_status()
    print(f"✅ Log cancellati per {agent}")


def print_logs(logs: list):
    if not logs:
        print("(nessun log trovato)")
        return

    for entry in reversed(logs):          # ordine cronologico
        print()
        print("─" * 70)
        print(f"[{entry['ts']}]  {entry['agent_code']}  |  {entry['sender_id']}  |  {entry['duration_ms']} ms")
        print(f"👤  {entry['user_msg']}")

        plan = entry.get("plan_json") or []
        if plan:
            print(f"📋  Piano ({len(plan)} task):")
            for t in plan:
                args = {k: v for k, v in (t.get("args") or {}).items()
                        if v is not None and v != "" and v != []}
                print(f"    [{t.get('id')}] {t.get('tool')}  deps={t.get('deps', [])}  args={args}")

        obs = entry.get("obs_json") or {}
        visible_obs = {k: v for k, v in obs.items() if k != "placeholder_map"}
        if visible_obs:
            print(f"🔍  Observations:")
            for k, v in visible_obs.items():
                v_str = json.dumps(v, ensure_ascii=False)
                if len(v_str) > 120:
                    v_str = v_str[:117] + "..."
                print(f"    {k}: {v_str}")

        response = entry.get("response", "")
        if response:
            lines = response.splitlines()
            preview = lines[0][:100] + ("…" if len(lines[0]) > 100 or len(lines) > 1 else "")
            print(f"🤖  {preview}")

    print("─" * 70)
    print(f"\n{len(logs)} turni mostrati.")


def main():
    parser = argparse.ArgumentParser(description="Fetch conversation logs")
    parser.add_argument("--agent",  default=DEFAULT_AGENT)
    parser.add_argument("--limit",  default=DEFAULT_LIMIT, type=int)
    parser.add_argument("--date",   default=None, help="es. 2026-02-20 (default: oggi)")
    parser.add_argument("--all-dates", action="store_true", help="nessun filtro data")
    parser.add_argument("--out",    default=None, help="salva JSON grezzo su file")
    parser.add_argument("--clear",  action="store_true", help="cancella log agente")
    parser.add_argument("--port",   default=9999, type=int)
    parser.add_argument("--host",   default="localhost")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    try:
        if args.clear:
            clear(base_url, args.agent)
            return

        date_filter = None
        if not args.all_dates:
            date_filter = args.date or str(date.today())

        logs = fetch(base_url, args.agent, args.limit, date_filter)

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            print(f"✅ {len(logs)} log salvati in {args.out}")
        else:
            print_logs(logs)

    except requests.exceptions.ConnectionError:
        print(f"❌ Server non raggiungibile su {base_url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ Errore HTTP: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
