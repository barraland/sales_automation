#!/usr/bin/env python3
"""
Client di test per il Beverage Sales Agent.

Uso:
    python test/chat_client.py                          # phone default AG001 (393755116724)
    python test/chat_client.py --phone 3925739617       # phone custom (AG002)
    python test/chat_client.py --port 8000

Comandi speciali durante la chat:
    /phone <numero>  — cambia numero di telefono simulato
    /reset           — cancella la chat history (nuovo thread)
    /list            — mostra ultimo risultato lista se presente
    /quit            — esci
"""

import argparse
import sys
import requests

DEFAULT_PHONE = "393755116724"  # AG001

# ── Helpers ───────────────────────────────────────────────────────────────────

def send_message(base_url: str, sender_id: str, text: str) -> dict:
    try:
        r = requests.post(
            f"{base_url}/test/chat",
            json={"sender_id": sender_id, "text": text},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"\n❌  Impossibile connettersi a {base_url}. Il server è avviato?")
        sys.exit(1)
    except requests.exceptions.Timeout:
        return {"text": "⏱️  Timeout: il server non ha risposto entro 120s."}
    except Exception as e:
        return {"text": f"❌  Errore: {e}"}


def reset_session(base_url: str, sender_id: str) -> None:
    try:
        r = requests.delete(
            f"{base_url}/test/chat",
            json={"sender_id": sender_id},
            timeout=10,
        )
        r.raise_for_status()
        print("🔄  Sessione resettata.\n")
    except Exception as e:
        print(f"❌  Errore reset: {e}\n")


def reset_all(base_url: str) -> None:
    """Chiama /admin/reset: ricrea DB ordini e svuota tutti i checkpoint."""
    try:
        r = requests.post(f"{base_url}/admin/reset", timeout=30)
        r.raise_for_status()
        data = r.json()
        print(
            f"🔄  Reset completo — clienti caricati: {data.get('clienti_caricati')}, "
            f"DB ordini: {data.get('db_ordini')}, checkpoint: {data.get('checkpoints')}\n"
        )
    except Exception as e:
        print(f"❌  Errore reset completo: {e}\n")


def print_response(resp: dict, last_list: list) -> list:
    """Stampa la risposta e ritorna la lista piatta (per selezione numerica)."""
    print()
    print("─" * 60)
    print(f"🤖  {resp.get('text', '')}")

    sections = resp.get("sections")
    items = resp.get("list")

    if sections:
        # Lista con sezioni per città
        flat = []
        print()
        print("📋  Lista interattiva:")
        for sec in sections:
            print(f"\n   📍 {sec['title']}")
            for item in sec.get("items", []):
                flat.append(item)
                desc = f" — {item['description']}" if item.get("description") else ""
                print(f"      {len(flat)}. {item['title']}{desc}")
        last_list = flat
        print()
        print("   Digita un numero per selezionare.")

    elif items:
        # Lista piatta (prodotti)
        print()
        print("📋  Lista interattiva:")
        for i, item in enumerate(items, 1):
            desc = f" — {item['description']}" if item.get("description") else ""
            print(f"   {i}. {item['title']}{desc}  [id: {item['id']}]")
        last_list = items
        print()
        print("   Digita un numero per selezionare.")

    print("─" * 60)
    print()
    # Se la risposta non contiene una lista, resetta last_list
    # (disambiguazione risolta → input numerici successivi = testo libero)
    if not sections and not items:
        return []
    return last_list


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chat client per Beverage Agent")
    parser.add_argument(
        "--phone",
        default=DEFAULT_PHONE,
        help=f"Numero di telefono simulato (sender_id). Default: {DEFAULT_PHONE} (AG001)",
    )
    parser.add_argument("--port", default=9999, type=int, help="Porta del server FastAPI")
    parser.add_argument("--host", default="localhost", help="Host del server FastAPI")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    sender_id = args.phone
    last_list: list = []

    print("=" * 60)
    print("  Beverage Sales Agent — Test Client")
    print("=" * 60)
    print(f"  Server  : {base_url}")
    print(f"  Phone   : {sender_id}")
    print()
    print("  Comandi: /phone <num>  /reset  /reset-all  /list  /quit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(f"[{sender_id}] Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nArrivederci!")
            break

        if not user_input:
            continue

        # ── Comandi speciali ──────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Arrivederci!")
            break

        if user_input.lower() == "/reset":
            reset_session(base_url, sender_id)
            last_list = []
            continue

        if user_input.lower() == "/reset-all":
            reset_all(base_url)
            last_list = []
            continue

        if user_input.lower() == "/list":
            if last_list:
                print("\n📋  Ultima lista ricevuta:")
                for i, item in enumerate(last_list, 1):
                    desc = f" — {item['description']}" if item.get("description") else ""
                    print(f"   {i}. {item['title']}{desc}  [id: {item['id']}]")
                print("   Digita un numero per selezionare.\n")
            else:
                print("   (nessuna lista nell'ultimo turno)\n")
            continue

        if user_input.lower().startswith("/phone"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                sender_id = parts[1].strip()
                print(f"✅  Phone cambiato: {sender_id}\n")
            else:
                print(f"   Uso: /phone <numero>  (attuale: {sender_id})\n")
            continue

        # Se l'utente scrive un numero intero, lo espande come selezione dalla lista
        if last_list and user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(last_list):
                item = last_list[idx]
                user_input = f"Ho selezionato: {item['title']} (ID: {item['id']})"
                print(f"   → espanso come: \"{user_input}\"")

        # ── Chiamata al server ────────────────────────────────────────────────
        print("   ⏳ ...")
        resp = send_message(base_url, sender_id, user_input)
        last_list = print_response(resp, last_list)


if __name__ == "__main__":
    main()
